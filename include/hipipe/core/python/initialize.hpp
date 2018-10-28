/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once
#include <hipipe/build_config.hpp>
#ifdef HIPIPE_BUILD_PYTHON

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL HIPIPE_PYTHON_ARRAY_SYMBOL

namespace hipipe::python {

/// \ingroup Python
/// \brief Initialize Python module, register OpenCV converters, exceptions, etc.
void initialize();

}  // namespace hipipe::python

#endif  // HIPIPE_BUILD_PYTHON
