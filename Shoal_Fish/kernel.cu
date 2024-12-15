#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>


#include <stdio.h>
#include <iostream>

#define NUM_BOIDS 5000 
#define BLOCK_SIZE 256
#define VISUAL_RANGE 50.0f
#define PROTECTED_RANGE 10.0f
#define AVOID_FACTOR 0.05f
#define MATCHING_FACTOR 0.05f
#define CENTERING_FACTOR 0.01f
#define MIN_SPEED 2.0f
#define MAX_SPEED 10.0f
#define DT 0.1f
#define TURN_FACTOR 1.0f
#define EDGE_MARGIN 50.0f
#define SCREEN_HEIGHT 600
#define SCREEN_WIDTH 800


struct BoidsVelocity
{
    float* vx, * vy;
};

const char* vertexShaderSource = R"(
#version 330 core
in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
})";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.1, 0.6, 0.9, 1.0); // Light blue
}
)";


BoidsVelocity boidsVelocity;
void toNormalised(float* x, float* y, float* norm_x, float* norm_y)
{
    *norm_x = (*x * 2) / SCREEN_WIDTH - 1.0f;
    *norm_y = 1.0f - (*y * 2) / SCREEN_HEIGHT;
}
void fromNormalised(float* x, float* y, float* norm_x, float* norm_y)
{
    *x = ((*norm_x + 1.0f) / 2.0f) * SCREEN_WIDTH;
    *y = (1.0f - ((*norm_y + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
}

__global__ void updateBoids(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numBoids) return;
    float close_dx = 0, close_dy = 0;
    float xvel_avg = 0, yvel_avg = 0, xpos_avg = 0, ypos_avg = 0;
    int neighbors = 0;

    float my_x = positions[2 * idx];
    float my_y = positions[2 * idx + 1];
    float my_vx = boidsVelocity.vx[idx];
    float my_vy = boidsVelocity.vy[idx];
    //fromNormalised(&my_x, &my_y, &positions[2 * idx], &positions[2 * idx + 1]);
    my_x = ((positions[2 * idx] + 1.0f) / 2.0f) * SCREEN_WIDTH;
    my_y = (1.0f - ((positions[2 * idx + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;

    // Loop through all boids
    for (int i = 0; i < numBoids; i++) {
        if (i == idx) continue;
        float x = ((positions[2 * i] + 1.0f) / 2.0f) * SCREEN_WIDTH;
        float y = (1.0f - ((positions[2 * i + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
        float dx = x - my_x;
        float dy = y - my_y;
        float dist = sqrt(dx * dx + dy * dy);

        if (dist < PROTECTED_RANGE) { // Separation
            close_dx -= dx;
            close_dy -= dy;
        }
        if (dist < VISUAL_RANGE) { // Alignment and Cohesion
            xvel_avg += boidsVelocity.vx[i];
            yvel_avg += boidsVelocity.vy[i];
            xpos_avg += x;
            ypos_avg += y;
            neighbors++;
        }
    }

    // Calculate alignment and cohesion
    if (neighbors > 0) {
        xvel_avg /= neighbors;
        yvel_avg /= neighbors;
        xpos_avg /= neighbors;
        ypos_avg /= neighbors;

        my_vx += (xvel_avg - my_vx) * MATCHING_FACTOR;   // Alignment
        my_vy += (yvel_avg - my_vy) * MATCHING_FACTOR;

        my_vx += (xpos_avg - my_x) * CENTERING_FACTOR;   // Cohesion
        my_vy += (ypos_avg - my_y) * CENTERING_FACTOR;
    }

    my_vx += close_dx * AVOID_FACTOR;  // Separation
    my_vy += close_dy * AVOID_FACTOR;

    // Edge Avoidance
    if (my_x < EDGE_MARGIN) my_vx += TURN_FACTOR;
    if (my_x > 800 - EDGE_MARGIN) my_vx -= TURN_FACTOR;
    if (my_y < EDGE_MARGIN) my_vy += TURN_FACTOR;
    if (my_y > 600 - EDGE_MARGIN) my_vy -= TURN_FACTOR;

    // Speed Limits
    float speed = sqrt(my_vx * my_vx + my_vy * my_vy);
    if (speed < MIN_SPEED) {
        my_vx = (my_vx / speed) * MIN_SPEED;
        my_vy = (my_vy / speed) * MIN_SPEED;
    }
    if (speed > MAX_SPEED) {
        my_vx = (my_vx / speed) * MAX_SPEED;
        my_vy = (my_vy / speed) * MAX_SPEED;
    }

    // Update position
    my_x += my_vx * dt;
    my_y += my_vy * dt;
    boidsVelocity.vx[idx] = my_vx;
    boidsVelocity.vy[idx] = my_vy;
    //fromNormalised(&my_x, &my_y, &positions[2 * idx], &positions[2 * idx + 1]);
    positions[2 * idx] = (my_x * 2) / SCREEN_WIDTH - 1.0f;
    positions[2 * idx + 1] = 1.0f - (my_y * 2) / SCREEN_HEIGHT;
}

void checkShaderCompilation(GLuint shader, std::string type) {
    GLint success;
    char infoLog[512];
    if (type == "PROGRAM") {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR: " << infoLog << std::endl;
        }
    }
    else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR (" << type << "): " << infoLog << std::endl;
        }
    }
}

void initBoids()
{

}

int main()
{
    cudaError_t cudaStatus;

    //int deviceCount = 0;
    //cudaGetDeviceCount(&deviceCount);
    //std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    //cudaDeviceReset();
    //cudaStatus = cudaGLSetGLDevice(0);  // Select the correct GPU
    //if (cudaStatus != cudaSuccess) {
    //    std::cerr << "Failed to set CUDA GL Device: " << cudaGetErrorString(cudaStatus) << std::endl;
    //    return -1;
    //}


    if (!glfwInit())
    {
        std::cout << "Failed to initialize the GLFW library" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Shoal of Fish", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, 800, 600);


    // Shader Compilation
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderCompilation(vertexShader, "VERTEX");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderCompilation(fragmentShader, "FRAGMENT");

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkShaderCompilation(shaderProgram, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    GLuint VBO, VAO;
    cudaGraphicsResource* cudaVBO;
    float* temp_positions = (float*)malloc(NUM_BOIDS * 2 * sizeof(float));

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);


    for (int i = 0; i < NUM_BOIDS * 2; i += 2) {
        //std::cout << cudaVBO << " ";

        temp_positions[i] = ((rand() % 800) / 400.0f) - 1.0f; // Normalize X to [-1, 1]
        temp_positions[i + 1] = ((rand() % 600) / 300.0f) - 1.0f; // Normalize Y to [-1, 1]
        /*((float*)cudaVBO)[i] = rand() % 800;
        ((float*)cudaVBO)[i + 1] = rand() % 600;*/
        //std::cout << &cudaVBO << " ";
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_BOIDS * 2 * sizeof(float), temp_positions, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error registering buffer with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc(&boidsVelocity.vx, NUM_BOIDS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }
    cudaStatus = cudaMalloc(&boidsVelocity.vy, NUM_BOIDS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }

    float* temp_vx = (float*)malloc(NUM_BOIDS * sizeof(float));
    float* temp_vy = (float*)malloc(NUM_BOIDS * sizeof(float));

    //cudaStatus = cudaGraphicsMapResources(1, &cudaVBO, 0);
    //if (cudaStatus != cudaSuccess) {
    //    std::cerr << "Error mapping CUDA resource!" << std::endl;
    //    return -1;  
    //}
    //size_t size;
    //cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&temp_positions, &size, cudaVBO);
    //if (cudaStatus != cudaSuccess) {
    //    std::cerr << "Error getting mapped pointer!" << std::endl;
    //    return -1;  
    //}
    //for (int i = 0; i < NUM_BOIDS * 2; i += 2) {
    //    //std::cout << cudaVBO << " ";
    //    
    //    ((float*)temp_positions)[i] = rand() % 800;
    //    ((float*)temp_positions)[i + 1] = rand() % 600;
    //    /*((float*)cudaVBO)[i] = rand() % 800;
    //    ((float*)cudaVBO)[i + 1] = rand() % 600;*/
    //    //std::cout << &cudaVBO << " ";
    //}
    //cudaStatus = cudaGraphicsUnmapResources(1, (cudaGraphicsResource**)&temp_positions, 0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaGraphicsUnmapResources failed!");
    //    return -1;
    //}
    for (int i = 0; i < NUM_BOIDS; i++) {
        temp_vx[i] = ((rand() % 20) - 10) / 10.0f;
        temp_vy[i] = ((rand() % 20) - 10) / 10.0f;
    }
    

    cudaStatus = cudaMemcpy(boidsVelocity.vx, temp_vx, NUM_BOIDS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }
    cudaStatus = cudaMemcpy(boidsVelocity.vy, temp_vy, NUM_BOIDS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }
    free(temp_vx);
    free(temp_vy);
    free(temp_positions);

    glPointSize(3.0f);
    while (!glfwWindowShouldClose(window))
    {

        cudaGraphicsMapResources(1, &cudaVBO, 0);
        cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&temp_positions, NULL, cudaVBO);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsResourceGetMappedPointer launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return -1;
        }

        // Update boids
        updateBoids << <(NUM_BOIDS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (temp_positions, boidsVelocity, NUM_BOIDS, DT);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();

        cudaStatus = cudaGraphicsUnmapResources(1, &cudaVBO, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsResourceGetMappedPointer launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return -1;
        }

        // Render
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_BOIDS);

        glfwSwapBuffers(window);
        glfwPollEvents();

    }



    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    cudaFree(boidsVelocity.vx);
    cudaFree(boidsVelocity.vy);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}




