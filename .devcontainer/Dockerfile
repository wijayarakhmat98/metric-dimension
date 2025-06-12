# check=skip=FromPlatformFlagConstDisallowed

FROM --platform=linux/amd64 debian:bookworm

ARG username="dev"

RUN \
	sed -i '/PS1.*\\\$/s/\\\$/\\n\0/' '/etc/skel/.bashrc' && \
	useradd "${username}" -ms '/bin/bash'

RUN \
	apt-get update && \
	apt-get -y dist-upgrade && \
	apt-get -y autopurge

RUN \
	apt-get -y install locales && \
	sed -i '/en_US.UTF-8/s/^# //g' '/etc/locale.gen' && \
	locale-gen

ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US:en"
ENV LC_ALL="en_US.UTF-8"

RUN \
	apt-get -y install \
		bzip2 \
		curl

USER "${username}"

RUN \
	curl -L 'https://micro.mamba.pm/api/micromamba/linux-64/latest' | tar -jxC "${HOME}" 'bin/micromamba' && \
	bash -lc 'micromamba shell init -s bash -r "${HOME}/micromamba"' && \
	bash -lc 'micromamba config append channels conda-forge' && \
	bash -lc 'micromamba config set channel_priority strict'

USER root

RUN \
	apt-get -y install \
		git \
		gnupg \
		less \
		man \
		openssh-client \
		patch \
		vim

USER "${username}"
