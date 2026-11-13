AUTODIFF_JULIA_CODE = '''

        using DifferentiationInterface
        import ForwardDiff
        using DifferentiationInterface: gradient, jacobian, derivative
        import GTPSA

        scibmad_native(x::AbstractArray) = Array(x)
        scibmad_native(x) = x

        function scibmad_native_args(args)
            return map(scibmad_native, args)
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

        function scibmad_vjp_arg(f, backend, args, arg_index, x, cotangent)
            native_args = scibmad_native_args(args)
            native_x = scibmad_native(x)
            native_cotangent = scibmad_native(cotangent)

            scalar_loss = function (y)
                output = scibmad_call_with_arg(
                    f, native_args, arg_index, y
                )

                return sum(vec(output) .* vec(native_cotangent))
            end

            return gradient(scalar_loss, backend, native_x)
        end
'''