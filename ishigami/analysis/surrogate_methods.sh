for n in 0.5 0.25 0.1 0.01 0.001 0.0001 0
do
  python ishigami/analysis/surrogate_methods.py --noise=$n
done

