for i in $(tail -n 45 ../files/longfiles.txt)
do
	echo "for $i$" 
	echo run number is below
	dasgoclient --query="run file=$i"
	echo "----------"
    echo
done
