FROM rockylinux:8

# RUN dnf install -y python38

# RUN pip install pytest

RUN apk add --no-cache python3 && python3 -m ensurepip && pip3 install pytest

ENTRYPOINT ["/bin/bash"]