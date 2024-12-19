#pragma once
#include "definitions.h"


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


void calculateTriangleVerticesCPU(float* positions, BoidsVelocity boidsVelocity, int num_boids);
int oneIterationCPU(float** boids_positions, BoidsVelocity* boidsVelocity);
void updateBoidsPositionCPU(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt);
void updateBoidsVelocityCPU(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt, bool cursorOverWindow, double cursorX, double cursorY);


__global__ void updateBoidsVelocity(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt, bool cursorOverWindow, double cursorX, double cursorY)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numBoids) return;
    float close_dx = 0, close_dy = 0;
    float xvel_avg = 0, yvel_avg = 0, xpos_avg = 0, ypos_avg = 0;
    int neighbors = 0;

    float my_x = positions[6 * idx];
    float my_y = positions[6 * idx + 1];
    float my_vx = boidsVelocity.vx[idx];
    float my_vy = boidsVelocity.vy[idx];
    my_x = ((positions[6 * idx] + 1.0f) / 2.0f) * SCREEN_WIDTH;
    my_y = (1.0f - ((positions[6 * idx + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;

    for (int i = 0; i < numBoids; i++) 
    {
        if (i == idx) continue;
        float x = ((positions[6 * i] + 1.0f) / 2.0f) * SCREEN_WIDTH;
        float y = (1.0f - ((positions[6 * i + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
        float dx = x - my_x;
        float dy = y - my_y;
        float dist = sqrt(dx * dx + dy * dy);

        // Separation
        if (dist < PROTECTED_RANGE) 
        { 
            close_dx -= dx;
            close_dy -= dy;
        }
        // Alignment and Cohesion
        if (dist < VISUAL_RANGE) 
        { 
            xvel_avg += boidsVelocity.vx[i];
            yvel_avg += boidsVelocity.vy[i];
            xpos_avg += x;
            ypos_avg += y;
            neighbors++;
        }
    }

    // Calculate alignment and cohesion
    if (neighbors > 0) 
    {
        xvel_avg /= neighbors;
        yvel_avg /= neighbors;
        xpos_avg /= neighbors;
        ypos_avg /= neighbors;

        // Alignment
        my_vx += (xvel_avg - my_vx) * MATCHING_FACTOR;   
        my_vy += (yvel_avg - my_vy) * MATCHING_FACTOR;

        // Cohesion
        my_vx += (xpos_avg - my_x) * CENTERING_FACTOR;   
        my_vy += (ypos_avg - my_y) * CENTERING_FACTOR;
    }

    // Separation
    my_vx += close_dx * AVOID_FACTOR;  
    my_vy += close_dy * AVOID_FACTOR;

    // Avoid cursor
    if (cursorOverWindow)
    {
        float dx_cursor = cursorX - my_x;
        float dy_cursor = cursorY - my_y;
        float dist_cursor = sqrtf(dx_cursor * dx_cursor + dy_cursor * dy_cursor);
        if (dist_cursor < VISUAL_RANGE)
        {
            my_vx -= dx_cursor * CURSOR_AVOID_FACTOR;
            my_vy -= dy_cursor * CURSOR_AVOID_FACTOR;
        }
    }

    // Bias 
    // biased to the right
    if (idx % 419 == 0)
    {
        my_vx = (1 - BIAS) * my_vx + (BIAS * 1);
    } 
    // biased to the left
    else if (idx % 409 == 0)
    {
        my_vx = (1 - BIAS) * my_vx + (BIAS * (-1));
    }
        
    // Edge Avoidance
    if (my_x < EDGE_MARGIN) my_vx += TURN_FACTOR;
    if (my_x > SCREEN_WIDTH - EDGE_MARGIN) my_vx -= TURN_FACTOR;
    if (my_y < EDGE_MARGIN) my_vy += TURN_FACTOR;
    if (my_y > SCREEN_HEIGHT - EDGE_MARGIN) my_vy -= TURN_FACTOR;

    // Speed Limits
    float speed = sqrt(my_vx * my_vx + my_vy * my_vy);
    if (speed < MIN_SPEED) 
    {
        my_vx = (my_vx / speed) * MIN_SPEED;
        my_vy = (my_vy / speed) * MIN_SPEED;
    }
    if (speed > MAX_SPEED)
    {
        my_vx = (my_vx / speed) * MAX_SPEED;
        my_vy = (my_vy / speed) * MAX_SPEED;
    }

    my_x += my_vx * dt;
    my_y += my_vy * dt;
    boidsVelocity.vx[idx] = my_vx;
    boidsVelocity.vy[idx] = my_vy;
}

__global__ void updateBoidsPosition(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numBoids) return;
    float my_x = ((positions[6 * idx] + 1.0f) / 2.0f) * SCREEN_WIDTH;
    float my_y = (1.0f - ((positions[6 * idx + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
    my_x += boidsVelocity.vx[idx] * dt;
    my_y += boidsVelocity.vy[idx] * dt;
    positions[6 * idx] = (my_x * 2) / SCREEN_WIDTH - 1.0f;
    positions[6 * idx + 1] = 1.0f - (my_y * 2) / SCREEN_HEIGHT;
}

void checkShaderCompilation(GLuint shader, std::string type)
{
    GLint success;
    char infoLog[512];
    if (type == "PROGRAM") 
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) 
        {
            glGetProgramInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR: " << infoLog << std::endl;
        }
    }
    else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) 
        {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR (" << type << "): " << infoLog << std::endl;
        }
    }
}

void processInput(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        Moving = !Moving;
    }
}

void cursorEnterCallback(GLFWwindow* window, int entered)
{
    if (entered) 
    {
        CursorOverWindow = true;
    }
    else {
        CursorOverWindow = false;
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    cursorX = xpos;
    cursorY = ypos;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

cudaError_t initBoids(GLuint* VBO, GLuint* VAO, cudaGraphicsResource** cudaVBO, BoidsVelocity* boidsVelocity)
{
    srand(time(NULL));
    cudaError_t cudaStatus;

    float* temp_positions = (float*)malloc(3 * NUM_BOIDS * 2 * sizeof(float));
    if (temp_positions == NULL)
    {
        fprintf(stderr, "malloc failed!");
        return cudaErrorMemoryAllocation;
    }
    float* temp_vx = (float*)malloc(NUM_BOIDS * sizeof(float));
    if (temp_vx == NULL)
    {
        fprintf(stderr, "malloc failed!");
        return cudaErrorMemoryAllocation;
    }
    float* temp_vy = (float*)malloc(NUM_BOIDS * sizeof(float));
    if (temp_vx == NULL)
    {
        fprintf(stderr, "malloc failed!");
        return cudaErrorMemoryAllocation;
    }

    glGenVertexArrays(1, VAO);
    glGenBuffers(1, VBO);
    glBindVertexArray(*VAO);

    for (int i = 0; i < 3 * NUM_BOIDS * 2; i += 2) 
    {
        temp_positions[i] = ((rand() % SCREEN_WIDTH) / (SCREEN_WIDTH / 2.0f)) - 1.0f; // Normalize X to [-1, 1]
        temp_positions[i + 1] = ((rand() % SCREEN_HEIGHT) / (SCREEN_HEIGHT / 2.0f)) - 1.0f; // Normalize Y to [-1, 1]
    }

    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
    glBufferData(GL_ARRAY_BUFFER, 3 * NUM_BOIDS * 2 * sizeof(float), temp_positions, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    cudaStatus = cudaGraphicsGLRegisterBuffer(cudaVBO, *VBO, cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) 
    {
        std::cerr << "Error registering buffer with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    cudaStatus = cudaMalloc(&boidsVelocity->vx, NUM_BOIDS * sizeof(float));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc(&boidsVelocity->vy, NUM_BOIDS * sizeof(float));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    for (int i = 0; i < NUM_BOIDS; i++) 
    {
        temp_vx[i] = ((rand() % 20) - 10) / 10.0f;
        temp_vy[i] = ((rand() % 20) - 10) / 10.0f;
    }

    cudaStatus = cudaMemcpy(boidsVelocity->vx, temp_vx, NUM_BOIDS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(boidsVelocity->vy, temp_vy, NUM_BOIDS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    free(temp_vx);
    free(temp_vy);
    free(temp_positions);

    return cudaSuccess;
}

void initShader(GLuint* shaderProgram)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderCompilation(vertexShader, "VERTEX");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderCompilation(fragmentShader, "FRAGMENT");

    *shaderProgram = glCreateProgram();
    glAttachShader(*shaderProgram, vertexShader);
    glAttachShader(*shaderProgram, fragmentShader);
    glLinkProgram(*shaderProgram);
    checkShaderCompilation(*shaderProgram, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

__global__ void calculateTriangleVertices(float* positions, BoidsVelocity boidsVelocity, int num_boids)
{
    int boid_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = boid_index * 6;
    if (boid_index >= num_boids) return;

    float my_x = ((positions[index] + 1.0f) / 2.0f) * SCREEN_WIDTH;
    float my_y = (1.0f - ((positions[index + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;

    // Boids are drawn as isosceles triangles of height 5*sqrt(3) heading in direction pointed by point (positions[index], positions[index + 1])
    float s =  2 * sqrtf(3);
    float triangle_h = 5 * sqrtf(3);



    float vector_length = sqrt(boidsVelocity.vx[boid_index] * boidsVelocity.vx[boid_index] + boidsVelocity.vy[boid_index] * boidsVelocity.vy[boid_index]);
    float h_x = my_x - (triangle_h * (boidsVelocity.vx[boid_index] / vector_length));
    float h_y = my_y - (triangle_h * (boidsVelocity.vy[boid_index] / vector_length));

    float x1 = h_x + ((my_y - h_y) / s);
    float x2 = h_x + ((h_y - my_y) / s);
    float y1 = h_y + ((h_x - my_x) / s);
    float y2 = h_y + ((my_x - h_x) / s);


    positions[index] = positions[index];
    positions[index + 2] = (x1 * 2) / SCREEN_WIDTH - 1.0f;
    positions[index + 4] = (x2 * 2) / SCREEN_WIDTH - 1.0f;

    positions[index + 1] = positions[index + 1];
    positions[index + 3] = 1.0f - (y1 * 2) / SCREEN_HEIGHT;
    positions[index + 5] = 1.0f - (y2 * 2) / SCREEN_HEIGHT;
}



cudaError oneIteration(cudaGraphicsResource** cudaVBO, float ** boids_positions, BoidsVelocity* boidsVelocity, int BLOCKS_NUM)
{
    cudaError cudaStatus;

    updateBoidsVelocity << <BLOCKS_NUM, BLOCK_SIZE >> > (*boids_positions, *boidsVelocity, NUM_BOIDS, DT, CursorOverWindow, cursorX, cursorY);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    updateBoidsPosition << <BLOCKS_NUM, BLOCK_SIZE >> > (*boids_positions, *boidsVelocity, NUM_BOIDS, DT);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    calculateTriangleVertices << <BLOCKS_NUM, BLOCK_SIZE >> > (*boids_positions, *boidsVelocity, NUM_BOIDS);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }


    return cudaSuccess;
}

void cleanUp(GLuint* VBO, GLuint* VAO, BoidsVelocity* boidsVelocity, GLuint* shaderProgram)
{
    glDeleteBuffers(1, VBO);
    glDeleteVertexArrays(1, VAO);
    cudaFree(boidsVelocity->vx);
    cudaFree(boidsVelocity->vy);
    glDeleteProgram(*shaderProgram);
    glfwTerminate();
}

int main()
{
    cudaError_t cudaStatus;
    int BLOCKS_NUM = (NUM_BOIDS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize the GLFW library\n");
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Shoal of Fish", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, processInput);
    glfwSetCursorEnterCallback(window, cursorEnterCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwTerminate();
        return -1;
    }

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    GLuint shaderProgram;
    initShader(&shaderProgram);

    GLuint VBO, VAO;
    cudaGraphicsResource* cudaVBO;
    BoidsVelocity boidsVelocity;
    float* boids_positions = NULL;


    cudaStatus = initBoids(&VBO, &VAO, &cudaVBO, &boidsVelocity);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Boids initialization failed: %s\n", cudaGetErrorString(cudaStatus));
        cleanUp(&VBO, &VAO, &boidsVelocity, &shaderProgram);
        return -1;
    }
    
    glPointSize(2.0f);

    int iterations = 0;
    clock_t start = clock();
    clock_t end;


    float* vx = NULL;
    float* vy = NULL;
    float* positions = NULL;
    BoidsVelocity boidsVelocityCPU;
    if (!gpuVersion)
    {
        vx = (float*)malloc(sizeof(float) * NUM_BOIDS);
        vy = (float*)malloc(sizeof(float) * NUM_BOIDS);
        cudaMemcpy(vx, boidsVelocity.vx, sizeof(float) * NUM_BOIDS, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy, boidsVelocity.vy, sizeof(float) * NUM_BOIDS, cudaMemcpyDeviceToHost);
        boidsVelocityCPU.vx = vx;
        boidsVelocityCPU.vy = vy;
        positions = (float*)malloc(sizeof(float) * NUM_BOIDS * 3 * 2);
    }

    while (!glfwWindowShouldClose(window))
    {

        if (Moving)
        {
            if (CursorOverWindow) 
            {
                glfwGetCursorPos(window, &cursorX, &cursorY);
            }
            cudaStatus = cudaGraphicsMapResources(1, &cudaVBO, 0);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaGraphicsMapResources launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return cudaStatus;
            }

            cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&boids_positions, NULL, cudaVBO);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaGraphicsResourceGetMappedPointer launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return cudaStatus;
            }
            if (gpuVersion)
            {
                cudaStatus = oneIteration(&cudaVBO, &boids_positions, &boidsVelocity, BLOCKS_NUM);
                if (cudaStatus != cudaSuccess)
                {
                    fprintf(stderr, "iteration launch failed: %s\n", cudaGetErrorString(cudaStatus));
                    cudaStatus = cudaGraphicsUnregisterResource(cudaVBO);
                    if (cudaStatus != cudaSuccess)
                    {
                        fprintf(stderr, "cudaGraphicsUnregisterResource launch failed: %s\n", cudaGetErrorString(cudaStatus));
                    }
                    cleanUp(&VBO, &VAO, &boidsVelocity, &shaderProgram);
                    return -1;
                }
            }
            else
            {
                cudaStatus = cudaMemcpy(positions, boids_positions, sizeof(float) * NUM_BOIDS * 3 * 2,  cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess)
                {
                    fprintf(stderr, "cudaMemcpy launch failed: %s\n", cudaGetErrorString(cudaStatus));
                }
                oneIterationCPU(&positions, &boidsVelocityCPU);
                cudaStatus = cudaMemcpy(boids_positions, positions, sizeof(float) * NUM_BOIDS * 3 * 2, cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess)
                {
                    fprintf(stderr, "cudaMemcpy launch failed: %s\n", cudaGetErrorString(cudaStatus));
                }
            }

            cudaStatus = cudaGraphicsUnmapResources(1, &cudaVBO, 0);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaGraphicsResourceGetMappedPointer launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return cudaStatus;
            }
        }
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3 * NUM_BOIDS);

        glfwSwapBuffers(window);
        glfwPollEvents();

        iterations++;
        if (iterations == ITERATIONS)
        {
            end = clock();
        }

    }

    if (iterations < ITERATIONS)
    {
        end = clock();
    }
    else
    {
        iterations = ITERATIONS;
    }
    float loopTime = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for %i iterations: %f\n", iterations, loopTime);

    cudaStatus = cudaGraphicsUnregisterResource(cudaVBO);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaGraphicsUnregisterResource launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cleanUp(&VBO, &VAO, &boidsVelocity, &shaderProgram);

    if (!gpuVersion)
    {
        free(boidsVelocityCPU.vx);
        free(boidsVelocityCPU.vy);
        free(positions);
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return -1;
    }

    return 0;
}


void updateBoidsVelocityCPU(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt, bool cursorOverWindow, double cursorX, double cursorY)
{
    for (int idx = 0; idx < numBoids; idx++)
    {
        float close_dx = 0, close_dy = 0;
        float xvel_avg = 0, yvel_avg = 0, xpos_avg = 0, ypos_avg = 0;
        int neighbors = 0;

        float my_x = positions[6 * idx];
        float my_y = positions[6 * idx + 1];
        float my_vx = boidsVelocity.vx[idx];
        float my_vy = boidsVelocity.vy[idx];
        my_x = ((positions[6 * idx] + 1.0f) / 2.0f) * SCREEN_WIDTH;
        my_y = (1.0f - ((positions[6 * idx + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;

        for (int i = 0; i < numBoids; i++)
        {
            if (i == idx) continue;
            float x = ((positions[6 * i] + 1.0f) / 2.0f) * SCREEN_WIDTH;
            float y = (1.0f - ((positions[6 * i + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
            float dx = x - my_x;
            float dy = y - my_y;
            float dist = sqrt(dx * dx + dy * dy);

            // Separation
            if (dist < PROTECTED_RANGE)
            {
                close_dx -= dx;
                close_dy -= dy;
            }
            // Alignment and Cohesion
            if (dist < VISUAL_RANGE)
            {
                xvel_avg += boidsVelocity.vx[i];
                yvel_avg += boidsVelocity.vy[i];
                xpos_avg += x;
                ypos_avg += y;
                neighbors++;
            }
        }

        // Calculate alignment and cohesion
        if (neighbors > 0)
        {
            xvel_avg /= neighbors;
            yvel_avg /= neighbors;
            xpos_avg /= neighbors;
            ypos_avg /= neighbors;

            // Alignment
            my_vx += (xvel_avg - my_vx) * MATCHING_FACTOR;
            my_vy += (yvel_avg - my_vy) * MATCHING_FACTOR;

            // Cohesion
            my_vx += (xpos_avg - my_x) * CENTERING_FACTOR;
            my_vy += (ypos_avg - my_y) * CENTERING_FACTOR;
        }

        // Separation
        my_vx += close_dx * AVOID_FACTOR;
        my_vy += close_dy * AVOID_FACTOR;

        // Avoid cursor
        if (cursorOverWindow)
        {
            float dx_cursor = cursorX - my_x;
            float dy_cursor = cursorY - my_y;
            float dist_cursor = sqrtf(dx_cursor * dx_cursor + dy_cursor * dy_cursor);
            if (dist_cursor < VISUAL_RANGE)
            {
                my_vx -= dx_cursor * CURSOR_AVOID_FACTOR;
                my_vy -= dy_cursor * CURSOR_AVOID_FACTOR;
            }
        }

        // Bias 
        // biased to the right
        if (idx % 419 == 0)
        {
            my_vx = (1 - BIAS) * my_vx + (BIAS * 1);
        }
        // biased to the left
        else if (idx % 409 == 0)
        {
            my_vx = (1 - BIAS) * my_vx + (BIAS * (-1));
        }

        // Edge Avoidance
        if (my_x < EDGE_MARGIN) my_vx += TURN_FACTOR;
        if (my_x > SCREEN_WIDTH - EDGE_MARGIN) my_vx -= TURN_FACTOR;
        if (my_y < EDGE_MARGIN) my_vy += TURN_FACTOR;
        if (my_y > SCREEN_HEIGHT - EDGE_MARGIN) my_vy -= TURN_FACTOR;

        // Speed Limits
        float speed = sqrt(my_vx * my_vx + my_vy * my_vy);
        if (speed < MIN_SPEED)
        {
            my_vx = (my_vx / speed) * MIN_SPEED;
            my_vy = (my_vy / speed) * MIN_SPEED;
        }
        if (speed > MAX_SPEED)
        {
            my_vx = (my_vx / speed) * MAX_SPEED;
            my_vy = (my_vy / speed) * MAX_SPEED;
        }

        my_x += my_vx * dt;
        my_y += my_vy * dt;
        boidsVelocity.vx[idx] = my_vx;
        boidsVelocity.vy[idx] = my_vy;
    }
}

void updateBoidsPositionCPU(float* positions, BoidsVelocity boidsVelocity, int numBoids, float dt)
{
    for (int idx = 0; idx < numBoids; idx++)
    {
        float my_x = ((positions[6 * idx] + 1.0f) / 2.0f) * SCREEN_WIDTH;
        float my_y = (1.0f - ((positions[6 * idx + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;
        my_x += boidsVelocity.vx[idx] * dt;
        my_y += boidsVelocity.vy[idx] * dt;
        positions[6 * idx] = (my_x * 2) / SCREEN_WIDTH - 1.0f;
        positions[6 * idx + 1] = 1.0f - (my_y * 2) / SCREEN_HEIGHT;
    }
}

int oneIterationCPU(float** boids_positions, BoidsVelocity* boidsVelocity)
{
    updateBoidsVelocityCPU(*boids_positions, *boidsVelocity, NUM_BOIDS, DT, CursorOverWindow, cursorX, cursorY);
    updateBoidsPositionCPU(*boids_positions, *boidsVelocity, NUM_BOIDS, DT);
    calculateTriangleVerticesCPU(*boids_positions, *boidsVelocity, NUM_BOIDS);
    return 0;
}



void calculateTriangleVerticesCPU(float* positions, BoidsVelocity boidsVelocity, int num_boids)
{
    for (int boid_index = 0; boid_index < num_boids; boid_index++)
    {
        int index = boid_index * 6;
        if (boid_index >= num_boids) return;

        float my_x = ((positions[index] + 1.0f) / 2.0f) * SCREEN_WIDTH;
        float my_y = (1.0f - ((positions[index + 1] + 1.0f) / 2.0f)) * SCREEN_HEIGHT;

        // Boids are drawn as isosceles triangles of height 5*sqrt(3) heading in direction pointed by point (positions[index], positions[index + 1])
        float s = 2 * sqrtf(3);
        float triangle_h = 5 * sqrtf(3);

        float vector_length = sqrt(boidsVelocity.vx[boid_index] * boidsVelocity.vx[boid_index] + boidsVelocity.vy[boid_index] * boidsVelocity.vy[boid_index]);
        float h_x = my_x - (triangle_h * (boidsVelocity.vx[boid_index] / vector_length));
        float h_y = my_y - (triangle_h * (boidsVelocity.vy[boid_index] / vector_length));

        float x1 = h_x + ((my_y - h_y) / s);
        float x2 = h_x + ((h_y - my_y) / s);
        float y1 = h_y + ((h_x - my_x) / s);
        float y2 = h_y + ((my_x - h_x) / s);

        positions[index] = positions[index];
        positions[index + 2] = (x1 * 2) / SCREEN_WIDTH - 1.0f;
        positions[index + 4] = (x2 * 2) / SCREEN_WIDTH - 1.0f;

        positions[index + 1] = positions[index + 1];
        positions[index + 3] = 1.0f - (y1 * 2) / SCREEN_HEIGHT;
        positions[index + 5] = 1.0f - (y2 * 2) / SCREEN_HEIGHT;
    }
}




