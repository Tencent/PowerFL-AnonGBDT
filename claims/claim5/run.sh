#!/bin/bash
set -e
# LAN
# for samples scalability
# anongbdt_main -r 0 -n 100000 -m 10 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 250000 -m 10 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 500000 -m 10 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 1000000 -m 10 -d 5 -B 16 -t 8


# for depth scalability
# anongbdt_main -r 0 -n 100000 -m 10 -d 3 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 10 -d 4 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 10 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 10 -d 6 -B 16 -t 8


# for feature scalability
# anongbdt_main -r 0 -n 100000 -m 10 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 25 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 50 -d 5 -B 16 -t 8

# anongbdt_main -r 0 -n 100000 -m 100 -d 5 -B 16 -t 8


function run_protocol() {
  local role=$1
  local setting=$2
  local command=""

  local n=100000
  local d=5
  local B=16
  local m=10
  local t=8
    
  case $setting in
    1)
        ;;
    2)
        n=250000
        ;;
    3)
        n=500000 
        ;;
    4)
        n=1000000
        ;;
    5)
        d=3
        ;;
    6)
        d=4
        ;;
    7)
        d=6
        ;;
    8)
        m=25
        ;;
    9)  
        m=50
        ;;
    10)
        m=100
        ;;
    *)
      echo "Invalid setting. "
      help
      exit 1
      ;;
  esac
  command="anongbdt_main -r ${role} -n ${n} -d ${d} -B ${B} -m ${m} -t ${t}"
  echo "running command: $command"
  $command
}


function help() {
  echo "run the whole protocol"
  echo "  ./run.sh -r [role] -s [setting]"
  echo "role: 0 for party 0, 1 for party 1"
  echo "setting 1: n = 10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 2: n = 2.5*10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 3: n = 5*10^5, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 4: n = 10^6, D = 5, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 5: n = 10^5, D = 3, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 6: n = 10^5, D = 4, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 7: n = 10^5, D = 6, B = 16, m0 = 10, m1 = 10, #threads = 8"
  echo "setting 8: n = 10^5, D = 5, B = 16, m0 = 25, m1 = 25, #threads = 8"
  echo "setting 9: n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50, #threads = 8"
  echo "setting 10: n = 10^5, D = 5, B = 16, m0 = 100, m1 = 100, #threads = 8"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--setting)
            setting="${2}"
            shift
            ;;
        -r|--role)
            role="${2}"
            shift
            ;;
        -h|--help)
            help
            exit 1
            ;;
        *)
            help
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$role" ]]; then help; exit 1; fi
if [[ -z "$setting" ]]; then help; exit 1; fi

run_protocol "${role}" "${setting}"
