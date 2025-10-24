FROM secretflow/ubuntu-base-ci:20240710 as base

FROM base as compile_base

WORKDIR /home/spu
COPY . .
RUN cd ./artifact \
    && apt-get update \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements-dev.txt \
    && ./install.sh

From base as runtime
COPY . .
COPY --from=compile_base /bin/anongbdt_main /usr/local/bin/
COPY --from=compile_base /bin/basegbdt_main /usr/local/bin/
COPY --from=compile_base /bin/benchmark /usr/local/bin/
COPY --from=compile_base /bin/anongbdt_inference /usr/local/bin/
RUN python3 -m pip install pandas
