#!/bin/bash
set -e

# evaluation argmax - n = 10^5, m = 10; LAN; 1 threads
benchmark -r 0 -p argmax -n 100000 -m 10

# evaluation greater - n = 10^6, LAN; 1 threads
benchmark -r 0 -p greater -n 1000000

# evaluation mux - n = 10^6, LAN; 1 threads
benchmark -r 0 -p mux -n 1000000

# evaluation binmatvec - n = 10^5, m = 128, LAN; 1 threads
benchmark -r 0 -p binmatvec -n 100000 -m 128

# evaluation sigmoid - n = 10^5, LAN; 1 threads
benchmark -r 0 -p sigmoid -n 100000

# evaluation fastpacklwes - m = 128, LAN; 1 threads
benchmark -p fastpacklwes -m 128

# evaluation fastpacklwes - m = 512, LAN; 1 threads
benchmark -p fastpacklwes -m 512

# evaluation bois - n = 10^6, LAN; 1 threads
benchmark -r 0 -p bois -n 1000000
