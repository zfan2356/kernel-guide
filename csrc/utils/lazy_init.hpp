#pragma once

#include <functional>
#include <memory>

#define K_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace kernels {

template <typename T> class LazyInit {
public:
    explicit LazyInit(std::function<std::shared_ptr<T>()> factory) : factory(std::move(factory)) {
    }

    T* operator->() {
        if (ptr == nullptr)
            ptr = factory();
        return ptr.get();
    }

private:
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;
};

} // namespace kernels
