========================================================================================
[FWR Architecture Master Specification — PERMANENTLY CLOSED]

System Type:          Deterministic Discrete-Time Dynamical System
External Inputs:      F(n) ∈ [0, 1],  C(n) ∈ [-1, 1],  W*_k ∈ [0, 1] (at crisis)
Internal State Space: X(n) = [T(n), W(n)]ᵀ ∈ [0, ∞) × [0, 1]
System Parameters:    α > 0 (Sat.), η ∈ (0, 1] (Eff.), γ ∈ (0, 1] (Decay), β ∈ [0, 1] (Compat.)
========================================================================================

1. State Vector & Output Equation:
   X(n) = [T(n), W(n)]ᵀ
   E(n) = min(F(n), W(n)) · C(n) · η · [ 1 + T(n) / (1 + αT(n)) ]

2. Crisis Predicate:
   crisis(n) :≡ E(n) ≤ γ · T(n)

3. Deterministic State Transition (n → n+1):
   If crisis(n) == True:
       T(n+1) = β · T(n)
       W(n+1) = W*_k   (External Decision)
   If crisis(n) == False:
       T(n+1) = T(n) + |E(n)| - γ · T(n)
       W(n+1) = W(n)

4. Derived Concepts & Convergence:
   Pivot Sequence = { n ∈ ℕ | crisis(n) == True }
   Stabilization  ⟺ ∃ K ∈ ℕ, ∀ n ≥ K, ¬crisis(n)
   Boundedness    ⟹ lim_{n→∞} T(n) ≤ (1 + 1/α) / γ
========================================================================================
