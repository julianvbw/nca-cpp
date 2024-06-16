//
// Created by julian on 02/03/24.
//
#pragma once
#include <glad/gl.h>

namespace gl {
    void init();

    void render(GLuint texture, GLuint vao, GLuint shader);

    void set_transform(GLint transform, float* mat);

    GLuint init_shaders();

    GLint init_transform(GLuint shader);

    GLuint init_vao();

    GLuint init_texture(unsigned width, unsigned height);
};
