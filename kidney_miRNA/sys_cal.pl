#!/usr/bin/perl
open(DATA1, "<gdc_sample_sheet.2024-04-08.tsv");
while($sukhen=<DATA1>)

 {
@a=split('\t',$sukhen);

if($sukhen=~/Primary Tumor/)
    {
    system("cp data/$a[0]/* Kidney_cancer_tumor_data");
    }
elsif($sukhen=~/Solid Tissue Normal/)
   {
   system("cp data/$a[0]/* Kidney_cancer_normal_data");
   }
else
   {
system("cp data/$a[0]/* not_matched");
    }

}

close(DATA1);
