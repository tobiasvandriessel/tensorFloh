#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

int extractFeaturesFromVideo(std::string path);

