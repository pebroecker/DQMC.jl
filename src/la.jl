function decompose_udt(M::Array{greens_type, 2}; full_U=false)
  Q, R, p = qr(M, Val{true}; thin=false)
  p_T = copy(p); p_T[p] = collect(1:length(p))
  D = abs(real(diag(triu(R))))
  T = (spdiagm(1./D) * R)[:, p_T]

  if full_U == true
    return Q, D, T
  else
    return Q[:, 1:length(D)], D, T
  end
end


function combine_udt(Ul, Dl, Tl, Ur, Dr, Tr)
  M = spdiagm(Dl) * (Tl * Ur) * spdiagm(Dr)
  Up, Dp, Tp = decompose_udt(M)
  return Ul * Up, Dp, Tp * Tr
end
