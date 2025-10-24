#!/bin/bash
set -e

# evaluation 1 - n = 5*10^4, D = 4, B = 8, m0 = 8, m1 = 7; LAN; 6 threads
# anongbdt_main -r 0 -n 50000 -d 4 -B 8 -m 8 -t 6

# evaluation 2 - n = 2*10^5, D = 4, B = 8, m0 = 8, m1 = 7; LAN; 6 threads
# anongbdt_main -r 0 -n 200000 -d 4 -B 8 -m 8 -t 6

# evaluation 3 - n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16, LAN; 8 threads
# anongbdt_main -r 0 -n 140000 -d 5 -B 10 -m 7 -t 8

# evaluation 4 - n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16; LAN 100mbps; 8 threads
# anongbdt_main -r 0 -n 140000 -d 5 -B 10 -m 7 -t 8

function run_protocol() {
  local role=$1
  local setting=$2
  local command=""
  case $setting in
    1)
      command="anongbdt_main -r ${role} -n 50000 -d 4 -B 8 -m 7 -t 6"
      ;;
    2)
      command="anongbdt_main -r ${role} -n 200000 -d 4 -B 8 -m 7 -t 6"
      ;;
    3)
      command="anongbdt_main -r ${role} -n 140000 -d 5 -B 10 -m 16 -t 8"
      ;;
    4)
      command="anongbdt_main -r ${role} -n 140000 -d 5 -B 10 -m 16 -t 8"
      ;;
    *)
      echo "Invalid setting. "
      help
      exit 1;
      ;;
  esac
  echo "running command: $command"
  $command
}


function help() {
  echo "run the whole protocol"
  echo "  ./run.sh -r [role] -s [setting]"
  echo "role: 0 for party 0, 1 for party 1"
  echo "setting 1: n = 5*10^4, D = 4, B = 8, m0 = 8, m1 = 7, #threads = 6, LAN"
  echo "setting 2: n = 2*10^5, D = 4, B = 8, m0 = 8, m1 = 7, #threads = 6, LAN"
  echo "setting 3: n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16, #threads = 8, LAN"
  echo "setting 4: n = 1.4*10^5, D = 5, B = 10, m0 = 7, m1 = 16, #threads = 8, WAN 100mbps (Please manually configure the traffic control via tc command)"
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
