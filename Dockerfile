# start by pulling the python image
FROM python:3.8

# copies all the files and directories in the current directory 
# (the directory where the Dockerfile is located) 
# into the root directory of the container's file system. 
COPY . /

# sets the working directory for any subsequent commands in the Dockerfile 
# to the root directory of the file system in the container. 
WORKDIR /

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8081
ENTRYPOINT [ "python" ]

# Run app.py when the container launches
CMD ["app.py"]