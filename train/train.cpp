#include <iostream>
#include <../common/utils.h>
#include <../common/model.h>
#include "single_step_lr.h"
#include "utils.h"
#include <tclap/CmdLine.h>
#include <filesystem>



int main(int argc, char** argv) {
    try {
    TCLAP::CmdLine cmd("Command-line interface for training the neural cellular automaton", ' ', "1.0");

    TCLAP::ValueArg<std::string>    image_arg ("i", "input",
                                               "Input reference image", true, "", "string", cmd);
    TCLAP::ValueArg<std::string>    output_arg("o", "output",
                                               "Training output path", false, "output", "string", cmd);
    TCLAP::ValueArg<unsigned> channel_arg ("c", "channels",
                                           "Number of channels (>3)", false, 16, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> padding_arg ("p", "padding",
                                           "Amount of zero-padding around the image", false, 4, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> poolsz_arg  ("P", "poolsize",
                                           "Size of training pool", false, 1024, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> batchsz_arg ("B", "batchsize",
                                           "Training batch size", false, 30, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> dmgsize_arg ("d", "holesize",
                                           "Diameter of damage augmentation", false, 4, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> dmgcount_arg("H", "holes",
                                           "Number of damage augmentations", false, 3, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> train_arg   ("e", "epochs",
                                           "Number of training epochs", false, 1000, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> miniter_arg ("m", "updates-min",
                                           "Minimum number of automaton grid updates", false, 64, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> maxiter_arg ("M", "updates-max",
                                           "Maximum number of automaton grid updates", false, 96, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> lrstep_arg  ("S", "learning-step",
                                           "Learning rate update after steps", false, 2000, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> growdeb_arg ("", "grow-steps",
                                           "Number of steps to test growing output. Requires verbosity >2",
                                           false, 400, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> traindeb_arg("", "train-output-interval",
                                            "Output training image every interval steps. Requires verbosity >2",
                                            false, 20, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> verbdeb_arg ("", "verbose-output-interval",
                                           "Output verbose output every interval steps. Requires verbosity >0",
                                           false, 100, "unsigned", cmd);
    TCLAP::ValueArg<float> lrgamma_arg("G", "learning-gamma",
                                       "Learning rate update factor", false, 0.1, "float", cmd);
    TCLAP::ValueArg<float> lrate_arg  ("R", "learning-rate",
                                       "Learning rate", false, 2e-3, "float", cmd);
    TCLAP::MultiSwitchArg verbose_arg("v", "verbose",
                                      "Verbosity, use -v for csv friendly output, -v -v for CLI loss output and "
                                      "-v -v -v for image sample outputs and growing evaluation", cmd);
    TCLAP::SwitchArg cuda_arg        ("C", "cpu-only",
                                      "Toggle cpu only usage", cmd, true);

    cmd.parse(argc, argv);

    /// Prepare file I/O variables
    const unsigned verbosity = verbose_arg.getValue();

    const std::filesystem::path ipath = image_arg.getValue();

    std::filesystem::path tmp = output_arg.getValue();
    const std::filesystem::path opath = (!tmp.has_filename()) ? tmp : tmp.parent_path();
    const std::string ofile = (!tmp.has_filename()) ? ("model-"+ipath.stem().string()+".lth") : tmp.filename().string();

    std::filesystem::create_directories(opath);
    if (verbosity > 2) {
        std::filesystem::create_directory(opath/"train-images");
        std::filesystem::create_directory(opath/"grow-images");
    }
    //

    /// Prepare LibTorch model
    const auto device = cuda_arg.getValue() ? torch::kCUDA : torch::kCPU;
    const unsigned c = channel_arg.getValue();

    if (c < 4) throw TCLAP::ArgException("Invalid number of channels (<4)");

    NCAModel model(c);
    model->to(device);
    //

    /// Load training image
    const unsigned pad = padding_arg.getValue();
    unsigned w, h;
    auto img_tensor = utils::load_image_to_tensor(ipath.c_str(), w, h).unsqueeze(0).to(device);
    w += 2*pad; h += 2*pad;

    // Zero-pad image tensor
    {
        using namespace torch::indexing;
        auto tmp_padded = torch::zeros({1, 4, w, h}).to(device);
        tmp_padded.index_put_({Slice(), Slice(), Slice(pad, w-pad), Slice(pad, h-pad)}, img_tensor);
        img_tensor = tmp_padded;
    }
    img_tensor.to(device);

    unsigned mx = w/2, my = h/2;
    //

    /// Prepare training
    const float lr = lrate_arg.getValue();
    const unsigned lr_step = lrstep_arg.getValue();
    const float gamma = lrgamma_arg.getValue();

    auto optim = torch::optim::Adam(model->update->parameters(), torch::optim::AdamOptions().lr(lr));
    auto lr_scheduler = torch::optim::StepLR(optim, lr_step, gamma);

    const unsigned steps = train_arg.getValue();

    const unsigned pool_size = poolsz_arg.getValue();
    const unsigned batch_size = batchsz_arg.getValue();
    const unsigned damage_count = dmgcount_arg.getValue();
    const unsigned damage_size = dmgsize_arg.getValue();

    auto pool = torch::zeros({pool_size, c, w, h}, device).detach();
    utils::seed(pool, 1, {mx, my}, {2, 3});
    trainutils::Batch batch;
    torch::Tensor sort_idx;

    /// Array of random update iterations for each training step
    const unsigned min_iters = miniter_arg.getValue();
    const unsigned max_iters = maxiter_arg.getValue();

    auto iters = torch::randint(min_iters, max_iters, {steps});

    // Print csv header
    if (verbosity == 1) std::cout << "step,loss" << std::endl;

    /// Start training
    model->train();
    for (unsigned step = 0; step < steps; ++step){
        batch = trainutils::sample_batch(batch_size, pool, device);
        sort_idx = trainutils::batch_argsort(batch, img_tensor);
        trainutils::batch_seed(batch, sort_idx, mx, my);
        trainutils::batch_randomdamage(batch, sort_idx, damage_count, damage_size);
        batch.tensor.requires_grad_();

        optim.zero_grad();

        for (unsigned i = 0; i < iters[step].item<int>(); ++i) {
            batch.tensor = model->forward(batch.tensor);
        }

        auto loss = torch::mse_loss(batch.tensor.slice(1, 0, 4), img_tensor.repeat({batch_size, 1, 1, 1}));
        loss.backward();

        model->normalize_grad_();

        optim.step();
        lr_scheduler.step();

        batch.tensor.detach_();
        trainutils::update_pool(batch, pool);

        // Print verbose output
        if (verbosity > 0) {
            const unsigned oint = verbdeb_arg.getValue();
            const unsigned oimgint = traindeb_arg.getValue();

            float progress = 100.0f * (float) step / (float) steps;
            if (step % ((steps/oint)+1) == 0) {
                if (verbosity == 1) {
                    std::cout << step << "," << loss.item<double>() << std::endl;
                }
                if (verbosity >  1)
                    std::cout << "\r\tProgress:\t"
                        << std::fixed << std::setprecision(1) << std::setfill('0')
                        << std::setw(4) << progress
                        << "%\t|\tLoss:\t"
                        << std::setprecision(6) << std::setw(8) << loss.item<double>() << std::flush;
                if ((verbosity > 2) && (step % ((steps/oimgint)+1) == 0)) {
                    std::filesystem::path path;
                    path = opath / "train-images" / (std::to_string(step / ((steps/oimgint)+1)) + ".png");
                    utils::save_tensor_as_image(
                        batch.tensor.slice(0, 0, 1).slice(1, 0, 4).squeeze(0).contiguous().to(torch::kCPU),
                        path.c_str()
                    );
                }
            }
        }
    }

    /// Optional grow test at the end of training
    if (verbosity > 2) {
        auto grid = torch::zeros({1, c, w, h}).to(device);
        model->eval();
        grid.detach_();
        grid.zero_();
        grid.slice(1, 3, c).slice(2, my, my + 1).slice(3, mx, mx + 1) = 1.0f;

        std::filesystem::path path;
        for (int i = 0; i < growdeb_arg.getValue(); ++i) {
            grid = model->forward(grid);
            path = opath / "grow-images" / (std::to_string(i) + ".png");
            utils::save_tensor_as_image(grid.slice(1, 0, 4).squeeze(0).clone().to(torch::kCPU), path.c_str()
            );
        }
    }

    /// Save the model
    model->to(torch::kCPU);
    torch::save(model, (opath / ofile).string());
    //


    } catch (TCLAP::ArgException &e)
    { std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}
