#Unsloth ile eğitim veya model çıktısı alınırken aşağıdaki kütüphaneler indirilmelidir
#Unsloth githubından aşağıdaki kod alınmıştır
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
