#!/usr/bin/perl
open(DATA1, "<renal_cell_carcinoma_NOS.csv");
open(DATA2, ">processed_renal_cell_carcinoma_NOS.csv");

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
print DATA2 "\n";
  }
close(DATA1);
close(DATA2);



