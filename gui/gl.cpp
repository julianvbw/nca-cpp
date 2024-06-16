//
// Created by julian on 02/03/24.
//

#include "gl.h"
#include <iostream>

const char *vertexShaderSource = "#version 330 core\n"
                                        "layout (location = 0) in vec3 pos_attr;\n"
                                        "layout (location = 1) in vec2 tex_attr;\n"
                                        "uniform mat3 transform;\n"
                                        "out vec2 tex_coord;\n"
                                        "void main()\n"
                                        "{\n"
                                        "   gl_Position = vec4(transform * pos_attr, 1.0);\n"
                                        "   tex_coord = tex_attr;\n"
                                        "}\0";


const char *fragmentShaderSource = "\n"
                                          "#version 330 core\n"
                                          "out vec4 frag_color;\n"
                                          "in vec2 tex_coord;\n"
                                          "uniform sampler2D image_tex;\n"
                                          "\n"
                                          "void main()\n"
                                          "{\n"
                                          "    vec4 tex_color = texture(image_tex, tex_coord);\n"
                                          "    if (tex_color.w < 0.1) discard;\n"
                                          "    frag_color = tex_color;\n"
                                          "}\0";

void gl::init(){
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void gl::render(GLuint texture, GLuint vao, GLuint shader){
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

GLuint gl::init_shaders(){
    GLuint vert_shader;
    vert_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert_shader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vert_shader);

    int success;
    char infoLog[512];
    glGetShaderiv(vert_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vert_shader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint frag_shader;
    frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag_shader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(frag_shader);

    glGetShaderiv(frag_shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(frag_shader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint shader_prog;
    shader_prog = glCreateProgram();

    glAttachShader(shader_prog, vert_shader);
    glAttachShader(shader_prog, frag_shader);
    glLinkProgram(shader_prog);

    glUseProgram(shader_prog);

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    return shader_prog;
}

GLint gl::init_transform(GLuint shader){
    return glGetUniformLocation(shader, "transform");
}

void gl::set_transform(GLint transform, float* mat){
    glUniformMatrix3fv(transform, 1, GL_TRUE, mat);
}

GLuint gl::init_vao(){
    const GLfloat vertices[] = {
            +0.5, +0.5, 0.0,    1.0, 0.0,
            -0.5, +0.5, 0.0,    0.0, 0.0,
            -0.5, -0.5, 0.0,    0.0, 1.0,
            +0.5, -0.5, 0.0,    1.0, 1.0,
    };

    const GLuint indices[] = { 0, 1, 2, 0, 2, 3};

    GLuint VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // vertices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*) 0);
    glEnableVertexAttribArray(0);

    // texture points
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return VAO;
}

GLuint gl::init_texture(unsigned width, unsigned height){
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei) width, (GLsizei) height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return texture;
}