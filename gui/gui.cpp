#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <lodepng.h>
#include <iostream>
#include <../common/model.h>
#include <../common/utils.h>
#include <tclap/CmdLine.h>
#include "cuda_interop.h"
#include "gl.h"

void framebuffer_size_callback(GLFWwindow* win, int w, int h){
    const int s = std::min(w, h);
    glViewport(std::max(w-s, 0)/2, std::max(h-s, 0)/2, s, s);
}


int main(int argc, char** argv){
    TCLAP::CmdLine cmd("Interactive neural cellular automaton UI", ' ', "1.0");

    TCLAP::UnlabeledValueArg<std::string> model_arg("model", "Trained libtorch model", true, "", "string", cmd);
    TCLAP::ValueArg<unsigned> dmgsize_arg("d", "holesize",
                                           "Diameter of damage augmentation", false, 4, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> height_arg("H", "height",
                                           "Canvas height", false, 32, "unsigned", cmd);
    TCLAP::ValueArg<unsigned> width_arg("W", "width",
                                           "Canvas width", false, 32, "unsigned", cmd);
    cmd.parse(argc, argv);

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(512, 512, "NCAGL", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwSetFramebufferSizeCallback(window, &framebuffer_size_callback);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    const unsigned width = width_arg.getValue();
    const unsigned height = height_arg.getValue();

    NCAModel model;
    torch::load(model, model_arg.getValue());

    const int channels = model->channels.item<int>();
    model = NCAModel(channels);
    torch::load(model, model_arg.getValue());

    model->to(torch::kCUDA);
    model->eval();

    auto grid = torch::zeros({1, channels, width, height}, torch::kCUDA);
    auto grid_img = torch::zeros({1, width, height, 4}, torch::TensorOptions().device(torch::kCUDA).layout(torch::kStrided).dtype(torch::kByte));
    utils::seed(grid, 1, {width/2, height/2}, {2, 3});

    gl::init();
    GLuint shader = gl::init_shaders();
    GLuint vao = gl::init_vao();
    GLuint texture = gl::init_texture(width, height);
    GLint transform = gl::init_transform(shader);

    auto mat = torch::eye(3, torch::kF32).mul(2.0f);
    gl::set_transform(transform, mat.data_ptr<float>());

    CudaTextureResource cudatex(texture);
    {
        double sx, sy, vx, vy;
        int vport[4];
        int state;
        double time, time_;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            if (state == GLFW_PRESS) {
                glfwGetCursorPos(window, &sx, &sy);
                glGetIntegerv(GL_VIEWPORT, &vport[0]);
                vx = (sx-vport[0])/vport[2];
                vy = (sy-vport[1])/vport[3];
                auto circ = utils::circle_mask((int) (vx*width), (int) (vy*height), 4, width, height, torch::kCUDA);
                grid.mul_(circ);
            }

            state = glfwGetKey(window, GLFW_KEY_R);
            if (state == GLFW_PRESS) {
                utils::seed(grid, 1, {width/2, height/2}, {2, 3});
            }

            time_ = glfwGetTime();
            if (time_ - time > 1.0f/30.0f){
                torch::NoGradGuard guard;
                grid = model->forward(grid);
                time = time_;
            }

            // copy data, so tensor is in correct layout
            grid_img.copy_(grid.slice(1, 0, 4).transpose(1, 3).clamp(0.0, 1.0).mul(255).to(torch::kByte));
            cudatex.set(grid_img);
            gl::render(texture, vao, shader);

            glfwSwapBuffers(window);
        }
    }

    glfwTerminate();
    return 0;
}
