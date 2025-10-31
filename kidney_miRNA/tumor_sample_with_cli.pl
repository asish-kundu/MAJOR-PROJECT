
#!/usr/local/bin/perl
open(DATA1,"<gdc_sample_sheet.2024-04-08.tsv");
open(DATA2,">sample_data_only_tumor_with_clinic.csv");
while($sukhen=<DATA1>)
 {
 chomp $sukhen;
@a=split('\t',$sukhen);
open(DATA3,"<clinical.tsv");
  while($sukhen2=<DATA3>)
      {
      @b=split('\t',$sukhen2);
      if($sukhen2=~/$a[5]/)
            {
           print DATA2 $sukhen."\t".$b[1]."\t".$b[9]."\t".$b[11]."\t".$b[15]."\t".$b[20]."\t".$b[27]."\t".$b[109]."\t".$b[127]."\n";
             }
          
      }
 close(DATA3);

 }
close(DATA1);
close(DATA2);
