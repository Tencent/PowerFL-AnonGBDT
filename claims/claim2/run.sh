#!/bin/bash
set -e
# evaluation 1 - n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50

# evaluation 1 - n = 2.5*10^5, D = 5, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 250000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 250000 -d 5 -B 16 -m 50

# evaluation 1 - n = 5*10^5, D = 5, B = 16, m0 = 50, m1 = 50, LAN; 1 threads
# basegbdt_main -r 0 -n 500000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 500000 -d 5 -B 16 -m 50

# evaluation 1 - n = 10^6, D = 5, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 1000000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 1000000 -d 5 -B 16 -m 50

# evaluation 2 - n = 10^5, D = 3, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 3 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 3 -B 16 -m 50

# evaluation 2 - n = 10^5, D = 4, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 4 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 4 -B 16 -m 50

# evaluation 2 - n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50, LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50

# evaluation 2 - n = 10^5, D = 6, B = 16, m0 = 50, m1 = 50; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 6 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 6 -B 16 -m 50


# evaluation 3 - n = 10^5, D = 5, B = 16, m0 = 10, m1 = 10; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 10
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 10

# evaluation 3 - n = 10^5, D = 5, B = 16, m0 = 25, m1 = 25; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 25
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 25

# evaluation 3 - n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50, LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 50

# evaluation 3 - n = 10^5, D = 5, B = 16, m0 = 100, m1 = 100; LAN; 1 threads
# basegbdt_main -r 0 -n 100000 -d 5 -B 16 -m 100
# anongbdt_main -r 0 -n 100000 -d 5 -B 16 -m 100

function run_protocol() {
  local role=$1
  local protocol=$2
  local setting=$3
  local command=""
  case $protocol in
    0)
      command="basegbdt_main"
      ;;
    1)
      command="anongbdt_main"
      ;;
    *)
      echo "Invalid protocol. "
      help
      exit 1
      ;;
  esac

  local n=100000
  local d=5
  local B=16
  local m=50
    
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
        m=10
        ;;
    9)  
        m=25
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
  command="${command} -r ${role} -n ${n} -d ${d} -B ${B} -m ${m}"
  echo "running command: $command"
  $command
}


function help() {
  echo "run the whole protocol"
  echo "  ./run.sh -r [role] -p [protocol] -s [setting]"
  echo "role: 0 for party 0, 1 for party 1"
  echo "protocol: 0 for Base AnonGBDT, 1 for Advanced AnonGBDT"
  echo "setting 1: n = 10^5, D = 5, B = 16, m0 = 50, m1 = 50, single thread"
  echo "setting 2: n = 2.5*10^5, D = 5, B = 16, m0 = 50, m1 = 50, single thread"
  echo "setting 3: n = 5*10^5, D = 5, B = 16, m0 = 50, m1 = 50, single thread"
  echo "setting 4: n = 10^6, D = 5, B = 16, m0 = 50, m1 = 50, single thread"

  echo "setting 5: n = 10^5, D = 3, B = 16, m0 = 50, m1 = 50, single thread"
  echo "setting 6: n = 10^5, D = 4, B = 16, m0 = 50, m1 = 50, single thread"
  echo "setting 7: n = 10^5, D = 6, B = 16, m0 = 50, m1 = 50, single thread"

  echo "setting 8: n = 10^5, D = 5, B = 16, m0 = 10, m1 = 10, single thread"
  echo "setting 9: n = 10^5, D = 5, B = 16, m0 = 25, m1 = 25, single thread"
  echo "setting 10: n = 10^5, D = 5, B = 16, m0 = 100, m1 = 100, single thread"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--setting)
            setting="${2}"
            shift
            ;;
        -p|--protocol)
            protocol="${2}"
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
if [[ -z "$protocol" ]]; then help; exit 1; fi
if [[ -z "$setting" ]]; then help; exit 1; fi

run_protocol "${role}" "${protocol}" "${setting}"
