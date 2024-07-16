## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.3
ARG PROTOC_VERSION=26.0

## Base Layer ##################################################################
FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} as base

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
    && dnf install -y make compat-openssl11 \
        procps \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

## Build #######################################################################
FROM rust:1.77-bullseye as build
ARG PROTOC_VERSION
ARG GITHUB_TOKEN

COPY tgis-tester /usr/src/tgis-tester

# Install protoc, no longer included in prost crate
RUN cd /tmp && \
    curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    unzip protoc-*.zip -d /usr/local && rm protoc-*.zip

# Get proto files from text-generation-inference repo
RUN mkdir -p /usr/src/proto && \
    git clone https://github.com/IBM/text-generation-inference.git /usr/src/text-generation-inference && \
    cp /usr/src/text-generation-inference/proto/generation.proto /usr/src/proto/generation.proto

WORKDIR /usr/src/tgis-tester

RUN cargo install --path .

## Tests ########################################################################
FROM base as tester

RUN useradd -u 2000 tgis -m -g 0

ENV HOME=/home/tgis \
    CONFIG=minimal \
    CONFIG_PATH=/app/config/${CONFIG}.yaml \
    IMAGE_TAG=6ac2ad7

RUN chmod -R g+rwx ${HOME}

COPY --from=build /usr/local/cargo/bin/tgis-tester /usr/local/bin/tgis-tester
COPY tests /app/tests
COPY tls /app/tls
COPY config /app/config

# Run as non-root user by default
USER tgis

CMD tgis-tester --config-path=/app/config/${CONFIG}.yaml --image-tag=${IMAGE_TAG}
