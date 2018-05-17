Alunos: Fábio Costa Farias Marques, Gustavo Henrique Fernandes Carvalho
Matriculas: 140039082, 140021671

A versão de Python utilizada foi a 3.5.2.

Para executar o programa nas 2 imagens do Requisito 1 navegue até a pasta que contém o programa "T4-Limiares.py" e execute o comando:
-> "python T4-Limiares.py"

Caso deseje executar o programa em toda a base de imagens de teste (Requisito 2) execute o comando:
-> "python T4-Limiares.py -f"


Para que o programa funcione, as imagens devem estar na hierarquia de pastas enviada no arquivo zip, tendo raiz na pasta "Images", que deve ficar na mesma pasta do programa. O diretório "Test", dentro da pasta GT, deve abrigar as imagens "Ground Truth" e a pasta "Test", dentro do diretório "ORI", deve abrigar as imagens originais, nesse caso para a base de imagens. Para avaliar o Requisito 1 existe a pasta "SkinDataset" dentro de "Images", sendo que dentro daquela também existem outras pastas "GT" e "ORI" e dentro de cada uma a pasta "Test". A mesma regra acima deve ser obedecida para correto funcionamento. (As imagens do Requisito 1 devem ficar de fato dentro da pasta "Test")

Para a correta utilização do programa, os nomes das imagens de toda a base de dados devem ser mudados para seu número e extensão, exemplo:

"img (243).jpg" -> "243.jpg"

O programa enviado, em suas últimas linhas possui uma função que realiza tal renomeação caso descomentada e pasta configurada.
