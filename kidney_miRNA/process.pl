#!/usr/bin/perl
open(DATA1, "<all.txt");
open(DATA2, ">all_processed.txt");

while($sukhen=<DATA1>)
   {
@a=split('\t',$sukhen);
$len=scalar @a;
print DATA2 $a[0];
$i=2;
while($i<$len)
   {
  print DATA2 "\t".$a[$i];
   $i=$i+4;
   }

  }
close(DATA1);
close(DATA2);



