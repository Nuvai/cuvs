/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(CUVS_C_EXPORTS)
    #define CUVS_API __declspec(dllexport)
  #else
    #define CUVS_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define CUVS_API __attribute__((visibility("default")))
#else
  #define CUVS_API
#endif
