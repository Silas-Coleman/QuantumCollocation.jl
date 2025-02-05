"""
This file includes chebyshev integrators for states and unitaries
"""

function nth_order_chebyshev(Gₜ::Matrix, n::Int)
    """
    Compute the nth order chebyshev approximation for the exponential of matrix Gₜ
    
    Inputs:
        Gₜ (Matrix): Matrix to exponetiate and approximate
        n (int): polynomial order of approximation
    
    Returns: nth order chebyshev approximation of the exponential of Gₜ
    """
    @assert n > 0
    Gₜ_powers = compute_powers_with_identity(Gₜ,n)
    C = sum(CHEBYSHEV_COEFFICIENTS[n] .* Gₜ_powers)
    return C
end

const CHEBYSHEV_COEFFICIENTS = OrderedDict{Int,Vector{Float64}}(
    # Each array consists of the sum of coefficients for Tᵢ up to n where each coefficient is ∫eˣ*Tᵢdx from -1 to 1
    0 => [1.266065877752008406176287280687,],
    1 => [1.266065877752008406176287280687,1.130318207984970069190922004054,],
    2 => [0.994570538217931843227859189938,1.130318207984970069190922004054,0.542990679068153125896856181498,],
    3 => [0.994570538217931843227859189938,0.997307658438978616999293080880,0.542990679068153125896856181498,0.177347399394655214077687332974,],
    4 => [1.000044778660025501793029434339,0.997307658438978616999293080880,0.499196755531403246752830682453,0.177347399394655214077687332974,0.043793923536749858327343787323,],
    5 => [1.000044778660025501793029434339,1.000022289998548252754062559688,0.499196755531403246752830682453,0.166488873156376337991702030195,0.043793923536749858327343787323,0.008686820990623100521843547028,],
    6 => [0.999999801337071225759700610070,1.000022289998548252754062559688,0.500006347344580603930808138102,0.166488873156376337991702030195,0.041635012034943692793564906651,0.008686820990623100521843547028,0.001439274334537444773388092933,],
    7 => [0.999999801337071225759700610070,0.999999900943311481604780510679,0.500006347344580603930808138102,0.166667985598270840252865809816,0.041635012034943692793564906651,0.008328596106834076917557752040,0.001439274334537444773388092933,0.000204699933593727390534686150,],
    8 => [1.000000000549551959494465336320,0.999999900943311481604780510679,0.499999972545199233842083685886,0.166667985598270840252865809816,0.041666886031850459970460320847,0.008328596106834076917557752040,0.001388275939486621106747077370,0.000204699933593727390534686150,0.000025499197525411785886662736,],
    9 => [1.000000000549551959494465336320,1.000000000274257061505522869993,0.499999972545199233842083685886,0.166666661185663783628640999268,0.041666886031850459970460320847,0.008333363992219500887559391344,0.001388275939486621106747077370,0.000198342753079829405352119132,0.000025499197525411785886662736,0.000002825413561732440023558429,],
    10 => [0.999999999998962385561185328697,1.000000000274257061505522869993,0.500000000074679595840621004754,0.166666661185663783628640999268,0.041666665796007272548617805796,0.008333363992219500887559391344,0.001388892599847544543495425629,0.000198342753079829405352119132,0.000024794442827213547775018834,0.000002825413561732440023558429,0.000000281901879279295869344359,],
    11 => [0.999999999998962385561185328697,0.999999999999481858914407439443,0.500000000074679595840621004754,0.166666666681168335051310691597,0.041666665796007272548617805796,0.008333333217393979613918375549,0.001388892599847544543495425629,0.000198413095538163710253656835,0.000024794442827213547775018834,0.000002755071103398142745317251,0.000000281901879279295869344359,0.000000025579075757926220431858,],
    12 => [1.000000000000001554312234475219,0.999999999999481858914407439443,0.499999999999860611499258311596,0.166666666681168335051310691597,0.041666666668895145642093069682,0.008333333217393979613918375549,0.001388888875525949714515028255,0.000198413095538163710253656835,0.000024801625447431996477091684,0.000002755071103398142745317251,0.000000275517327974006745975607,0.000000025579075757926220431858,0.000000002128183768429711374033,],
    13 => [1.000000000000001554312234475219,1.000000000000000666133814775094,0.499999999999860611499258311596,0.166666666666640123084519586882,0.041666666668895145642093069682,0.008333333333619567795391702703,0.001388888875525949714515028255,0.000198412697050430204132043244,0.000024801625447431996477091684,0.000002755735249620650265735570,0.000000275517327974006745975607,0.000000025047758779920335784357,0.000000002128183768429711374033,0.000000000163482147078733886824,],
    14 => [1.000000000000000222044604925031,1.000000000000000666133814775094,0.500000000000000111022302462516,0.166666666666640123084519586882,0.041666666666662695306388997096,0.008333333333619567795391702703,0.001388888888920665172846402591,0.000198412697050430204132043244,0.000024801587176816665034854298,0.000002755735249620650265735570,0.000000275573458209825541669210,0.000000025047758779920335784357,0.000000002087361778743318040276,0.000000000163482147078733886824,0.000000000011663425624683783565,],
    15 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.500000000000000111022302462516,0.166666666666666685170383743753,0.041666666666662695306388997096,0.008333333333332837086770972235,0.001388888888920665172846402591,0.000198412698415816930340091484,0.000024801587176816665034854298,0.000002755731912008673902935840,0.000000275573458209825541669210,0.000000025052128017416455915747,0.000000002087361778743318040276,0.000000000160569322081320104006,0.000000000011663425624683783565,0.000000000000776753332643675539,],
    16 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.499999999999999944488848768742,0.166666666666666685170383743753,0.041666666666666671292595935938,0.008333333333332837086770972235,0.001388888888888835382307007649,0.000198412698415816930340091484,0.000024801587301862275565522697,0.000002755731912008673902935840,0.000000275573191445854909622447,0.000000025052128017416455915747,0.000000002087677045254063087077,0.000000000160569322081320104006,0.000000000011469415464225377609,0.000000000000776753332643675539,0.000000000000048502540114601584,],
    17 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.499999999999999944488848768742,0.166666666666666657414808128124,0.041666666666666671292595935938,0.008333333333333333217685101602,0.001388888888888835382307007649,0.000198412698412693343881160746,0.000024801587301862275565522697,0.000002755731922420641330360351,0.000000275573191445854909622447,0.000000025052108329332828429643,0.000000002087677045254063087077,0.000000000160590524632919136025,0.000000000011469415464225377609,0.000000000000764637588872796681,0.000000000000048502540114601584,0.000000000000002850763240206795,],
    18 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.499999999999999944488848768742,0.166666666666666657414808128124,0.041666666666666664353702032031,0.008333333333333333217685101602,0.001388888888888888941894328433,0.000198412698412693343881160746,0.000024801587301586864496788856,0.000002755731922420641330360351,0.000000275573192241482962555100,0.000000025052108329332828429643,0.000000002087675695097377277926,0.000000000160590524632919136025,0.000000000011470750784024637229,0.000000000000764637588872796681,0.000000000000047790369554996079,0.000000000000002850763240206795,0.000000000000000158260124356779,],
    19 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.499999999999999944488848768742,0.166666666666666657414808128124,0.041666666666666664353702032031,0.008333333333333333217685101602,0.001388888888888888941894328433,0.000198412698412698439631371428,0.000024801587301586864496788856,0.000002755731922398554946260696,0.000000275573192241482962555100,0.000000025052108385552393681103,0.000000002087675695097377277926,0.000000000160590438141282182583,0.000000000011470750784024637229,0.000000000000764716666940869256,0.000000000000047790369554996079,0.000000000000002811224206170516,0.000000000000000158260124356779,0.000000000000000008324007165532,],
    20 => [1.000000000000000222044604925031,1.000000000000000000000000000000,0.499999999999999944488848768742,0.166666666666666657414808128124,0.041666666666666664353702032031,0.008333333333333333217685101602,0.001388888888888888941894328433,0.000198412698412698439631371428,0.000024801587301587301565789639,0.000002755731922398554946260696,0.000000275573192239856500477694,0.000000025052108385552393681103,0.000000002087675698793833609237,0.000000000160590438141282182583,0.000000000011470745584613355794,0.000000000000764716666940869256,0.000000000000047794789054585540,0.000000000000002811224206170516,0.000000000000000156180359844093,0.000000000000000008324007165532,0.000000000000000000415952902537,]
    )

function compute_powers_with_identity(G::AbstractMatrix{T}, order::Int) where T <: Number
    powers = Array{typeof(G)}(undef, order+1)
    powers[1] = G^0
    powers[2] = G
    for k = 2:order
        powers[k+1] = powers[k] * G
    end
    return powers
end

function chebyshev_operator(
    G_powers::Vector{<:AbstractMatrix},
    coeffs::Vector{<:Real}
)
    return sum(coeffs .* G_powers)
end

function chebyshev_coefficients(Δt::Real,n::Int;timestep_derivative=false)
    if !timestep_derivative
        return CHEBYSHEV_COEFFICIENTS[n] .* ((Δt .^ (0:n)))
    else
        coeffs = CHEBYSHEV_COEFFICIENTS[n] .* ((Δt .^ (-1:n-1))) .* (1:n)
        coeffs[1] *= 0
        return coeffs
    end
end

function chebyshev_operator(
    G_powers::Vector{<:AbstractMatrix},
    n::Int,
    Δt::Int,
    timestep_derivative=false
)
    chebyshev_coefficients = chebyshev_coefficients(Δt,n;timestep_derivative)
    return chebyshev_operator(G_powers,chebyshev_coefficients)
end


# ----------------------------------------------------------------
#                    Quantum Chebyshev Integrator
# ----------------------------------------------------------------



abstract type QuantumChebyshevIntegrator <: QuantumIntegrator end



# ----------------------------------------------------------------
#                    Unitary Chebyshev Integrator
# ----------------------------------------------------------------

struct UnitaryChebyshevIntegrator <: QuantumChebyshevIntegrator
    unitary_component::Vector{Int}
    drive_components::Vector{Int}
    timestep::Union{Real, Int} # either the timestep or the index of the timestep
    freetime::Bool
    n_drives::Int
    ketdim::Int
    dim::Int
    zdim::Int
    order::Int
    autodiff::Bool
    G::Function
    ∂G::Function

    function UnitaryChebyshevIntegrator(
        sys::AbstractQuantumSystem,
        unitary_name::Symbol,
        drive_name::Union{Symbol,Tuple{Vararg{Symbol}}},
        traj::NamedTrajectory;
        order::Int=4,
        G::Function=a ->G_bilinear(a, sys.G_drift, sys.G_drives),
        ∂G::Function=a ->sys.G_drives,
        autodiff::Bool=false
    )
        @assert 0<order<=20
        ketdim = size(sys.H_drift,1)
        dim = 2ketdim^2

        unitary_components = traj.components[unitary_name]

        if drive_name isa Tuple
            drive_components = vcat((traj.components[s] for s ∈ drive_name)...)
        else
            drive_components = traj.components[drive_name]
        end

        n_drives = length(drive_components)

        @assert all(diff(drive_components) .== 1) "controls must be in order"

        freetime = traj.timestep isa Symbol

        if freetime
            timestep = traj.components[traj.timestep][1]
        else
            timestep = traj.timestep
        end

        return new(
            unitary_components,
            drive_components,
            timestep,
            freetime,
            n_drives,
            ketdim,
            dim,
            traj.dim,
            order,
            autodiff,
            G,
            ∂G
        )
    end
end

# ------------------- Integrator -------------------

function nth_order_chebyshev(
    QCI::QuantumChebyshevIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = QCI.G(aₜ)

    C = chebyshev_operator(Gₜ,QCI.order,Δt)

    I_N = sparse(I)

    return Ũ⃗ₜ₊₁ - (I_N ⊗ C) * Ũ⃗ₜ
end

@views function(UCI::UnitaryChebyshevIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
) # Go to _integrator_utils?

    Ũ⃗ₜ₊₁ = zₜ₊₁[UCI.unitary_components]
    Ũ⃗ₜ = zₜ[UCI.unitary_components]
    aₜ = zₜ[UCI.drive_components]

    if UCI.freetime
        Δtₜ = zₜ[UCI.timestep]
    else
        Δtₜ = UCI.timestep
    end
    
    return nth_order_chebyshev(UCI, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
end

# ------------------- Jacobians -------------------

function ∂aₜ(
    UCI::UnitaryChebyshevIntegrator,
    G_powers::Vector{<:AbstractMatrix},
) # TODO
end

function ∂Δtₜ(
    UCI::UnitaryChebyshevIntegrator,
    Gₜ_powers::Vector{<:AbstractMatrix},
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    Δtₜ::Real
)
    coeffs = chebyshev_coefficients(Δtₜ,UCI.order;timestep_derivative=true)
    ∂ΔtₜC=sum(coeffs .* Gₜ_powers)

    I_N = sparse(I, UCI.ketdim, UCI.ketdim)

    return Ũ⃗ₜ₊₁ - (I_N ⊗ ∂ΔtₜC) * U⃗̃ₜ
end

# ----------------------------------------------------------------
#               Quantum State Chebyshev Integrator
# ----------------------------------------------------------------

struct QuantumStateChebyshevIntegrator <: QuantumChebyshevIntegrator
    state_components::Vector{Int}
    drive_components::Vector{Int}
    timestep::Union{Real,Int}
    freetime::Bool
    n_drives::Int
    ketdim::Int
    dim::Int
    zdim::Int
    order::Int
    autodiff::Bool
    G::Function
    ∂G::Function

    function QuantumStateChebyshevIntegrator(
        sys::AbstractQuantumSystem,
        state_name::Symbol,
        drive_name::Union{Symbol,Tuple{Vararg{Symbol}}},
        traj::NamedTrajectory;
        order::Int=4,
        G::Function=a -> G_bilinear(a,sys.G_drift,sys.G_drives),
        ∂G::Function=a ->sys.G_drives,
        autodiff::Bool=false,
    )
        @assert 0<order<=20

        ketdim = size(sys.H_drift, 1)
        dim = 2ketdim

        state_components = traj.components[state_name]

        if drive_name isa Tuple
            drive_components = vcat((traj.components[s] for s ∈ drive_name)...)
        else
            drive_components = traj.components[drive_name]
        end

        n_drives = length(drive_components)

        @assert all(diff(drive_components) .== 1) "controls must be in order"

        freetime = traj.timestep isa Symbol
        
        if freetime
            timesep = traj.compoents[traj.timestep][1]
        else
            timestep = traj.stimestep
        end

        return new(
            state_components,
            drive_components,
            timestep,
            freetime,
            n_drives,
            ketdim,
            dim,
            traj.dim,
            order,
            autodiff,
            G,
            ∂G
        )
    end
end

function get_comps(QSCI::QuantumStateChebyshevIntegrator, traj::NamedTrajectory)
    if QSCI.freetime
        return QSCI.state_components, QSCI.drive_components, traj.compoents
    else
        return QSCI.state_components, QSCI.drive_components
    end
end

function (integrator::QuantumStateChebyshevIntegrator)(
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory;
    state_name::Union{Symbol, Nothing}=nothing,
    drive_name::Union{Symbol, Tuple{Vararg{Symbol}}, Nothing}=nothing,
    order::Int=integrator.order,
    G::Function=integrator.G,
    ∂G::Function=integrator.∂G,
    autodiff::Bool=integrator.autodiff
)
    @assert !isnothing(state_name)
    @assert !isnothing(drive_name)
    return QuantumStateChebyshevIntegrator(
        sys,
        state_name,
        drive_name,
        traj;
        order=order,
        G=G,
        ∂G=∂G,
        autodiff=autodiff
    )
end

# ------------------- Integrator -------------------

function nth_order_chebyshev(
    QSCI::QuantumStateChebyshevIntegrator,
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = QSCI.G(aₜ)

    C = chebyshev_operator(Gₜ,QSCI.order,Δt)

    return ψ̃ₜ₊₁ - C * ψ̃ₜ
end

@views function(QSCI::QuantumStateChebyshevIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
)
    ψ̃ₜ₊₁ = zₜ₊₁[QSCI.state_components]
    ψ̃ₜ = zₜ[QSCI.state_components]
    aₜ = zₜ[QSCI.drive_components]

    if QSCI.freetime
        Δtₜ = zₜ[QSCI.timestep]
    else
        Δtₜ = QSCI.timestep
    end
    return nth_order_chebyshev(QSCI, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)
end

function ∂Δtₜ(
    QSCI::QuantumStateChebyshevIntegrator,
    Gₜ_powers::Vector{<:AbstractMatrix},
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    Δtₜ::Real
)
    coeffs = chebyshev_coefficients(Δtₜ, QSCI.order; timestep_derivative=true)
    ∂ΔtₜC = sum(coeffs .* Gₜ_powers)

    return ψ̃ₜ₊₁ - ∂ΔtₜC * ψ̃ₜ
end