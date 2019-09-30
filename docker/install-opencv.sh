# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

DEBIAN_FRONTEND=noninteractive

 apt-get -y update
#  apt-get -y upgrade       # Uncomment to install new versions of packages currently installed
#  apt-get -y dist-upgrade  # Uncomment to handle changing dependencies with new vers. of pack.
#  apt-get -y autoremove    # Uncomment to remove packages that are now no longer needed


# 2. INSTALL THE DEPENDENCIES

# Build tools:
apt-get install -y build-essential cmake apt-utils gnupg2 software-properties-common curl wget

# Python:
# python-numpy  python3-numpy
add-apt-repository -y ppa:jonathonf/python-3.6 && apt-get -y update

apt-get install -y python3.6 python3.6-dev
curl https://bootstrap.pypa.io/get-pip.py | python3.6 - --user
ln -f -s /usr/bin/python3.6 /usr/local/bin/python3

alias pip="python3.6 -m pip"


apt-get install -y libopencv-dev
# apt-get install -y --allow-unauthenticated python-dev  python-tk  pylint  \
#                         python3-dev python3-tk pylint3 flake8 \
#                         python-pip python3-pip
