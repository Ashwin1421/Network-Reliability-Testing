# Network Reliability Testing

A network reliability testing program which uses exhaustive enumeration to get exact reliability metric for small complete graphs. 

## Getting Started

On GitHub, navigate to the main page of the repository.
Under the repository name, click **Clone or download.**

### Prerequisites

python 3.6
networkx
matplotlib
numpy

### Installing

After cloning the repository, simply launch a local terminal window and traverse to your local directory.
(Command Prompt in case of Windows/ Terminal in case of Linux). 
Run the main program by executing the following commands :-

Normal Usage : Pass <N> no. of nodes and <p> probability value.
```
python Main.py <N> <p>
```
Run the program over intervals of <p> in range [p1,p2] with interval of p1.

```
python Main.py -i <N> <p1> <p2>
```
Run the program for multiple values of <k> for a fixed value of <p>.
Values of k are in range [k1, k2]. 
```
python Main.py -k <N> <p> <k1> <k2>
```

Examples : 
```
python Main.py 5 0.05
python Main.py -i 5 0.05 1
python Main.py -k 5 0.9 0 20

```

## Author

**Ashwin Joshi**
**ashwin.joshi12@live.com**
