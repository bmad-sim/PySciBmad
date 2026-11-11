AUTODIFF_JULIA_CODE = '''

        using DifferentiationInterface
        import ForwardDiff
        using DifferentiationInterface: gradient, jacobian, derivative

        scibmad_native(x::AbstractArray) = Array(x)
        scibmad_native(x) = x

        function scibmad_native_args(args)
            return map(scibmad_native, args)
        end

        const scibmad_backends = Dict{String, Any}(
            "forwarddiff" => AutoForwardDiff()
        )

        # Still a work in progress!!!
        function scibmad_register_optional_backends!()
            candidates = (
                ("reversediff", :ReverseDiff, AutoReverseDiff),
                ("mooncake", :Mooncake, AutoMooncake),
                ("gtpsa", :GTPSA, AutoGTPSA),
                # ("enzyme", :Enzyme, AutoEnzyme), Enzyme not compiling figure this out later
                # ("zygote", :Zygote, AutoZygote), FIgure out zyfote later too
            )

            for (name, package, constructor) in candidates
                try
                    Base.eval(Main, Meta.parse("import $(package)"))
                    scibmad_backends[name] = constructor()
                catch err
                    @warn "Backend $name unavailable" exception=(
                        err, catch_backtrace()
                    )
                end
            end

            return collect(keys(scibmad_backends))
        end

        const scibmad_active_backend = Ref("forwarddiff")

        function scibmad_set_backend(name::AbstractString)
            key = lowercase(String(name))

            haskey(scibmad_backends, key) ||
                throw(ArgumentError("Unknown AD backend: $name"))

            scibmad_active_backend[] = key
            return key
        end

        function scibmad_get_backend()
            return scibmad_backends[scibmad_active_backend[]]
        end

        scibmad_get_backend_name() = scibmad_active_backend[]

        function scibmad_get_tracking_backend()
            name = scibmad_active_backend[]

            if name == "reversediff"
                @warn "ReverseDiff is not currently compatible with SciBmad tracking; using ForwardDiff for tracking gradients."
                return scibmad_backends["forwarddiff"]
            end
            
            return scibmad_get_backend()
        end

        function scibmad_register_backend(name::AbstractString, ad_backend)
            scibmad_backends[lowercase(String(name))] = ad_backend
            return nothing
        end

        function scibmad_bind_kwargs(f, kwargs)
            nt = (; (Symbol(k) => v for (k, v) in kwargs)...)

            return function (args...)
                return f(args...; nt...)
            end
        end

        function scibmad_call_with_arg(f, args, arg_index, x)
            """
            Call f with args, substituting x for the argument at arg_index

            Arguments:
            - f: the function to call (typically one of the element-tracking functions)
            - args: the full tuple of arguments f was originally called with
            - arg_index: the 1-based position within args to substitute x into
            - x: the value to substitute at arg_index

            Returns:
            - result: the result of calling f with x substituted in at arg_index
            """
                    return f((
                i == arg_index ? x : args[i]
                for i in eachindex(args)
            )...)
        end
        

        function scibmad_derivative_arg(f, backend, args, arg_index, x)
            """
            Compute the derivative of f with respect to a single scalar argument holding all other arguments fixed, 
            used when both the differentiation variable and f's output are scalar

            Arguments:
            - f: the function to differentiate
            - backend: the AutoForwardDiff() backend to use
            - args: the full tuple of arguments f was originally called with
            - arg_index: the 1-based position of the argument to differentiate with respect to
            - x: the value of that argument to differentiate at

            Returns:
            - deriv: the scalar derivative of f with respect to argument arg_index, evaluated at x
            """
            native_args = scibmad_native_args(args)
            native_x = scibmad_native(x)

            return derivative(
                y -> scibmad_call_with_arg(f, native_args, arg_index, y),
                backend,
                native_x
            )
        end


        function scibmad_gradient_arg(f, backend, args, arg_index, x)
            """
            Compute the gradient of f with respect to a vector-valued argument holding all other arguments fixed,
            used when the differentiation variable is a vector but f's output is scalar

            Arguments:
            - f: the function to differentiate
            - backend: the AutoForwardDiff() backend to use
            - args: the full tuple of arguments f was originally called with
            - arg_index: the 1-based position of the argument to differentiate with respect to
            - x: the vector value of that argument to differentiate at

            Returns:
            - grad: the gradient vector of f with respect to argument arg_index, evaluated at x
            """
            native_args = scibmad_native_args(args)
            native_x = scibmad_native(x)

            return gradient(
            y -> scibmad_call_with_arg(f, native_args, arg_index, y),
            backend,
            native_x
            )
        end

        
        # POSSIBLY REVERT THIS CHANGE?
        #=
        function scibmad_jacobian_arg(f, backend, args, arg_index, x)
            """
            Compute the full Jacobian of f with respect to a vector-valued argument holding all other arguments fixed,
            used when both the differentiation variable and f's output are vector-valued

            Arguments:
            - f: the function to differentiate
            - backend: the AutoForwardDiff() backend to use
            - args: the full tuple of arguments f was originally called with
            - arg_index: the 1-based position of the argument to differentiate with respect to
            - x: the vector value of that argument to differentiate at

            Returns:
            - jac: the Jacobian matrix of f with respect to argument arg_index, evaluated at x (output size x input size)
            """

            return jacobian(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end =#

        function scibmad_jacobian_arg(f, backend, args, arg_index, x) 
            native_args = scibmad_native_args(args)
            native_x = scibmad_native(x)

            return jacobian(
                y -> scibmad_call_with_arg(
                    f, native_args, arg_index, y
                ),
                backend,
                native_x
            )
        end

        function scibmad_jacobian_scalar_arg(f, backend, args, arg_index, x)
            """
            Compute the Jacobian of f with respect to a scalar argument (treated as a length-1 vector so the same Jacobian machinery can be reused),
            used when the differentiation variable is scalar but f's output is vector-valued

            Arguments:
            - f: the function to differentiate
            - backend: the AutoForwardDiff() backend to use
            - args: the full tuple of arguments f was originally called with
            - arg_index: the 1-based position of the argument to differentiate with respect to
            - x: the scalar value of that argument to differentiate at (internally wrapped as [x])

            Returns:
            - jac: the Jacobian of f with respect to the scalar argument, evaluated at x, shape (output size, 1)
            """
            native_args = scibmad_native_args(args)
            native_x = scibmad_native(x)

            return jacobian(
                y -> scibmad_call_with_arg(
                    f, native_args, arg_index, y[1]
                ),
                backend,
                [native_x]
            )
        end

        # Fix Jacobian erros w functions beliow 

        function _torch_flatten(x)
            if ndims(x) == 1
                return collect(x)
            elseif ndims(x) == 2 
                return vec(permutedims(x))
            else
                error("AD currently supports 1D and 2D arrays")
            end
        end 

        function _torch_unflatten(x_flat, original)
            if ndims(original) == 1 
                return reshape(x_flat, size(original))
            elseif ndims(original) == 2 
                return permutedims(reshape(x_flat, reverse(size(original))))
            else
                error("AD currently supports 1D and 2D arrays")
            end
        end
'''