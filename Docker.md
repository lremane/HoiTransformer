# Setup
## build image
docker build -t hoi-transformer:<tag> . 

## remove old container
docker stop <container-name>
docker rm <container-name>

## deploy new image
docker run -d hoi-transformer:<tag>