# EMDLoss_PyTorch_cpp_extension

This is not my original code. The orignal code's url is https://github.com/justanhduc/neuralnet-pytorch
I just convert format of this code to PyTorch cpp extension format


# Setup
Before you start, nvidia-docker, docker-compose are should be installed in your enviroment.
After installing, build dockerfile using below commands in root folder

## Environment
| Module list | Version |
| :--------:  | :--------:|
| Ubuntu      | 20.04 |
| CUDA        | 11.3.1 |
| CuDNN       | 8-dev |
| python3     | 3.9 |
| Pytorch     | 1.10.1+cu113 |

## Build dockerfile and Start docker container
~~~
docker-compose build
docker-compose up -d
docker attach emd_loss
~~~

## Run virtual environment in docker container
~~~
source /opt/venv/bin/activate
~~~

## Build & Evaluate emd_loss function
~~~
python setup.py install
python test.py
~~~

# Contact
if you have any questions please feel free to contact me
E-mail : yym064@naver.com
