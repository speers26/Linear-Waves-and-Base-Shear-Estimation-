

Directional Spectrum

nT = [same as time version]
nTheta = 32
Theta = np.linspace(from = 0, by=11.25, to=348.75)

Defines
S = [nT, nTheta]

S wrapped normal distribution in Theta dimension producted with JONSWAP in Freq/Dimension (Code from Jake)

A, B  = randn( nT, nTheta)  *  S * dtheta * dt

This is bigger than time only e.g. nTheta=1

Adjust A, B as time case for conditional waves (no change)

Z = A + iB

k_x = cos()/sin stuff

k = k_x * x + k_y * y

Z = sum(exp(1i * k) .* Z,2)

eta = fftshift(real(fft(Z,Spec.nf,1)),1