#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "definitions.h"

#include <stdio.h>
#include <iostream>

#define ITERATIONS 1000
#define NUM_BOIDS 10000
#define BLOCK_SIZE 256
#define VISUAL_RANGE 50.0f
#define PROTECTED_RANGE 10.0f
#define AVOID_FACTOR 0.05f
#define CURSOR_AVOID_FACTOR 5.0f
#define MATCHING_FACTOR 0.05f
#define CENTERING_FACTOR 0.001f
#define BIAS 0.2f
#define MIN_SPEED 2.0f
#define MAX_SPEED 10.0f
#define DT 0.4f
#define TURN_FACTOR 0.15f
#define EDGE_MARGIN 70.0f
#define SCREEN_HEIGHT 900
#define SCREEN_WIDTH 1800


struct BoidsVelocity
{
    float* vx, * vy;
};



bool gpuVersion = true;
bool Moving = true;
bool CursorOverWindow = false;
double cursorX;
double cursorY;



