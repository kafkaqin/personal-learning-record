z = 1+2j
print(z)
print(type(z))

z = 1 + 2j

real_part = z.real
print(f"实部: {real_part}")
imag_part = z.imag
print(f"虚部: {imag_part}")

modulus = abs(z)
print(f"模: {modulus}")

conj = z.conjugate()
print(f"共轭: {conj}")

z1 = 1 + 2j
z2 = 3-4j
print(f"加法: {z1 + z2}")
print(f"减法: {z1 - z2}")
print(f"乘法: {z1 * z2}")
print(f"除法: {z1 / z2}")
print(f"平方: {z1 **2}")

if isinstance(z,complex):
    print("This is complex")

def complex_calculator(z: complex):
    print(f"复数: {z}")
    print(f"实部: {z.real}")
    print(f"虚部: {z.imag}")
    print(f"模: {abs(z):.4f}")
    print(f"共轭: {z.conjugate()}")
    print(f"平方: {z**2}")
    print(f"倒数: {1/z}")

# 使用示例
z = 1 + 2j
complex_calculator(z)

import cmath
z = 1 + 2j
print(cmath.phase(z))
print(abs(z))
print(cmath.polar(z))
print(cmath.rect(2.236,1.107))
print(cmath.exp(z))
print(cmath.log(z))
