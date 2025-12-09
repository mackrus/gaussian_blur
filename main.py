import numpy as np
from PIL import Image


img = Image.open("photos/nice_dog.jpg").convert("L")

# B är bildmatrisen
B = np.asarray(img)

# p är en sklaningsfaktor till kärnan
p = 10

# Simpel matris
S = (1 / p**2) * np.ones((p, p))

# K är kärnan i konvolutionen, vi väljer S som kärna
K = S

# Bilden är en (mxk)-matris
m = np.max(B.shape)

# Kärnan är en (nxn)-matris
n = K.shape[0]

# vi lägger in padding för att kärnan ska ha samma dimension som bilden
padding_dim = np.max(B.shape)

padding_rows_b = padding_dim - m
padding_cols_b = padding_dim - m

padding_rows_k = padding_dim - n
padding_cols_k = padding_dim - n


# casehantering för när k>n eller när n>k, fallet lika med täcks också eftersom differensen blir 0
if B.shape[0] >= B.shape[1]:
    B_padded = np.pad(
        B,
        ((0, padding_rows_b), (0, padding_cols_b + abs((B.shape[0] - B.shape[1])))),
        "constant",
        constant_values=0,
    )
else:
    B_padded = np.pad(
        B,
        ((0, padding_rows_b + abs((B.shape[0] - B.shape[1]))), (0, padding_cols_b)),
        "constant",
        constant_values=0,
    )


K_padded = np.pad(
    K, ((0, padding_rows_k), (0, padding_cols_k)), "constant", constant_values=0
)


# vi fouriertransformerar både bilden och kärnan
B_f = np.fft.fft2(B_padded)
K_f = np.fft.fft2(K_padded)


# produkten av deras fouriertransformer inverstransformeras
B_output = np.fft.ifft2(B_f * K_f)

# ta bort imaginärdelen
B_output = np.real(B_output)

output = Image.fromarray(B_output)
img.show()
output.show()
