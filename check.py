from CartoonGan import CartoonGAN

all = CartoonGAN('')
generator = all.generator()
print(generator.summary())

descriminator = all.discriminator()
print(descriminator.summary())