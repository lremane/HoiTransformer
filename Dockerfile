FROM python:3.9-slim
CMD ["/bin/bash"]


# Install necessary tools and clean up
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    wget \
#    sudo \
#    && rm -rf /var/lib/apt/lists/*
#
## Set the working directory
#WORKDIR /HoiTransformer
#
## Copy project files
#COPY . /HoiTransformer
#
## (Optional) Install Python dependencies
## RUN pip install -r requirements.txt
#
## Default command
#CMD ["/bin/bash"]
