FROM debian:sid

RUN apt-get update && \
    apt-get -y install locales && \
    sed -i '/en_US.UTF-8/s/^# //g' '/etc/locale.gen' && \
    locale-gen

ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN apt-get -y dist-upgrade && \
    apt-get -y autopurge && \
    apt-get -y install curl git gnupg less man patch ssh vim

RUN apt-get -y install bzip2 jq nauty xz-utils

COPY --chmod=644 '.vimrc' '/etc/skel'
ARG username='dev'
RUN sed -i '/PS1.*\\\$/s/\\\$/\\n\0/' '/etc/skel/.bashrc' && \
    useradd "${username}" -ms '/bin/bash'

USER "${username}"

COPY 'environment.yml' '/tmp'
RUN export SHELL="$(getent passwd "${username}" | cut -d: -f7)" && \
    export BIN_FOLDER="${HOME}/bin" && \
    curl -L 'micro.mamba.pm/install.sh' | sh -s && \
    . ~/.profile && \
    micromamba env -y create -f '/tmp/environment.yml' -n 'z3' && \
	  printf "\nmicromamba activate 'z3'\n" >> "${HOME}/.bashrc"

USER root

ARG workspaceFolder
RUN install -d -o "${username}" -g "$(id -gn "${username}")" "${workspaceFolder}" && \
    ln -s "${workspaceFolder}/host/.vscode" "${workspaceFolder}/.vscode"

USER "${username}"
