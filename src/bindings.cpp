#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "core/TensorImpl.h"
#include "data/CSVLoader.h"
#include "loss/CrossEntropyLoss.h"
#include "loss/MSELoss.h"
#include "nn/Linear.h"
#include "nn/Serialization.h"
#include "nn/activations/ReLU.h"
#include "nn/activations/Sigmoid.h"
#include "optimizers/Adam.h"
#include "optimizers/AdamW.h"
#include "optimizers/RMSProp.h"
#include "optimizers/SGD.h"
#include "optimizers/SGDWithMomentum.h"

namespace py = pybind11;

PYBIND11_MODULE(axon, module) {
    module.doc() = "AXON";

    py::class_<TensorImpl, Tensor>(module, "Tensor")
        .def_static("from_data", &TensorImpl::from_data,
            py::arg("data"), py::arg("shape"), py::arg("require_grad") = false)
        .def_static("zeros", &TensorImpl::zeros,
            py::arg("shape"), py::arg("require_grad") = false)
        .def_static("ones", &TensorImpl::ones,
            py::arg("shape"), py::arg("require_grad") = false)
        .def_static("full", &TensorImpl::full,
            py::arg("shape"), py::arg("value"), py::arg("require_grad") = false)
        .def_static("randn", &TensorImpl::randn,
            py::arg("shape"), py::arg("require_grad") = false)
        .def_static("rand", &TensorImpl::rand,
            py::arg("shape"), py::arg("require_grad") = false)
        .def_static("eye", &TensorImpl::eye,
            py::arg("n"), py::arg("require_grad") = false)
        .def_static("arange", &TensorImpl::arange,
            py::arg("start"), py::arg("end"), py::arg("step") = 1.0f, py::arg("require_grad") = false)
        .def_static("copy", &TensorImpl::copy, py::arg("tensor"))

        // Shape and metadata
        .def_property_readonly("shape", &TensorImpl::get_shape)
        .def_property_readonly("strides", &TensorImpl::get_strides)
        .def_property_readonly("ndim", &TensorImpl::ndim)
        .def_property_readonly("nelem", &TensorImpl::nelem)
        .def("size", &TensorImpl::size, py::arg("dim"))
        .def_property("require_grad",
            &TensorImpl::get_require_grad, &TensorImpl::set_require_grad)
        .def("is_contiguous", &TensorImpl::is_contiguous)

        // Data access
        .def("item", [](const TensorImpl &t) {
            if (t.nelem() != 1) throw std::runtime_error("item() requires a scalar tensor");
            return t.at(0);
        })
        .def("__getitem__", [](const TensorImpl &t, int idx) { return t.at(idx); })
        .def("__setitem__", [](TensorImpl &t, int idx, float val) { t.at(idx) = val; })
        .def("tolist", [](const TensorImpl &t) {
            std::vector<float> out(t.nelem());
            const float *d = t.data();
            for (int i = 0; i < t.nelem(); ++i) out[i] = d[i];
            return out;
        })

        // Autograd
        .def("backward", &TensorImpl::backward)
        .def("grad", &TensorImpl::grad)
        .def("has_grad", &TensorImpl::has_grad)
        .def("zero_grad", &TensorImpl::zero_grad)

        // Operators
        .def("__add__", [](const Tensor &a, const Tensor &b) { return a + b; })
        .def("__sub__", [](const Tensor &a, const Tensor &b) { return a - b; })
        .def("__mul__", [](const Tensor &a, const Tensor &b) { return a * b; })
        .def("__truediv__", [](const Tensor &a, const Tensor &b) { return a / b; })
        .def("__neg__", [](const Tensor &a) { return -a; })
        .def("__add__", [](const Tensor &a, float s) { return a + s; })
        .def("__radd__", [](const Tensor &a, float s) { return s + a; })
        .def("__sub__", [](const Tensor &a, float s) { return a - s; })
        .def("__rsub__", [](const Tensor &a, float s) { return s - a; })
        .def("__mul__", [](const Tensor &a, float s) { return a * s; })
        .def("__rmul__", [](const Tensor &a, float s) { return s * a; })
        .def("__truediv__", [](const Tensor &a, float s) { return a / s; })
        .def("__iadd__", [](Tensor &a, const Tensor &b) { a += b; return a; })
        .def("__isub__", [](Tensor &a, const Tensor &b) { a -= b; return a; })
        .def("__imul__", [](Tensor &a, const Tensor &b) { a *= b; return a; })
        .def("__itruediv__", [](Tensor &a, const Tensor &b) { a /= b; return a; })

        // reprs
        .def("__repr__", [](const TensorImpl &t) {
            std::string s = "Tensor(shape=[";
            for (int i = 0; i < t.ndim(); ++i) {
                if (i) s += ", ";
                s += std::to_string(t.get_shape()[i]);
            }
            s += "], data=[";
            int n = std::min(t.nelem(), 6);
            for (int i = 0; i < n; ++i) {
                if (i) s += ", ";
                s += std::to_string(t.at(i));
            }
            if (t.nelem() > 6) s += ", ...";
            s += "])";
            return s;
        });

    // Tensor fns
    module.def("exp", [](const Tensor &t) { return exp(t); });
    module.def("log", [](const Tensor &t) { return log(t); });
    module.def("sqrt", [](const Tensor &t) { return sqrt(t); });
    module.def("abs", [](const Tensor &t) { return abs(t); });
    module.def("pow", [](const Tensor &t, float e) { return pow(t, e); }, py::arg("tensor"), py::arg("exponent"));
    module.def("pow", [](const Tensor &a, const Tensor &b) { return pow(a, b); });
    module.def("clip", [](const Tensor &t, float lo, float hi) { return clip(t, lo, hi); });
    module.def("matmul", [](const Tensor &a, const Tensor &b) { return matmul(a, b); });

    py::class_<Module>(module, "Module")
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters)
        .def("zero_grad", &Module::zero_grad);

    py::class_<Linear, Module>(module, "Linear")
        .def(py::init<int, int, bool>(),
            py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    py::class_<ReLU, Module>(module, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);

    py::class_<Sigmoid, Module>(module, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward);

    py::class_<Loss>(module, "Loss")
        .def("forward", &Loss::forward);

    py::class_<MSELoss, Loss>(module, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward);

    py::class_<CrossEntropyLoss, Loss>(module, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward);

    py::class_<Optimizer>(module, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad);

    py::class_<SGD, Optimizer>(module, "SGD")
        .def(py::init<std::vector<Tensor>, float>(),
            py::arg("parameters"), py::arg("lr") = 1e-3f);

    py::class_<SGDWithMomentum, Optimizer>(module, "SGDWithMomentum")
        .def(py::init<std::vector<Tensor>, float, float>(),
            py::arg("parameters"), py::arg("lr") = 1e-3f, py::arg("momentum") = 0.9f);

    py::class_<Adam, Optimizer>(module, "Adam")
        .def(py::init<std::vector<Tensor>, float, float, float, float>(),
            py::arg("parameters"), py::arg("lr") = 1e-3f,
            py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("epsilon") = 1e-8f);

    py::class_<AdamW, Optimizer>(module, "AdamW")
        .def(py::init<std::vector<Tensor>, float, float, float, float, float>(),
            py::arg("parameters"), py::arg("lr") = 1e-3f,
            py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
            py::arg("epsilon") = 1e-8f, py::arg("weight_decay") = 1e-2f);

    py::class_<RMSProp, Optimizer>(module, "RMSProp")
        .def(py::init<std::vector<Tensor>, float, float, float>(),
            py::arg("parameters"), py::arg("lr") = 1e-3f,
            py::arg("decay_rate") = 0.9f, py::arg("epsilon") = 1e-8f);

    module.def("load_csv", &axon::data::load_csv,
        py::arg("path"), py::arg("label_col"), py::arg("header") = true);

    module.def("save", &axon::save, py::arg("params"), py::arg("path"));
    module.def("load", &axon::load, py::arg("params"), py::arg("path"));
}
