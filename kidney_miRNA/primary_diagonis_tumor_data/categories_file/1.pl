#!/usr/bin/perl
# Open the data file for reading
open(DATA1, "<sample_data_only_tumor_with_clinic.csv") or die "Cannot open file: $!";

# Process each line in the file
while($sukhen = <DATA1>) {

    # Split the line into an array @a using tab as delimiter
    @a = split('\t', $sukhen);

    # Check the line for specific keywords and copy files accordingly
    if($sukhen =~ /Clear cell adenocarcinoma, NOS/) {
        system("cp Kidney_cancer_tumor_data/$a[1] clear_cell_adenocarcinoma");
    }
    elsif($sukhen =~ /Renal cell carcinoma, NOS/) {
        system("cp Kidney_cancer_tumor_data/$a[1] renal_cell_carcinoma_NOS");
    }
    elsif($sukhen =~ /Papillary adenocarcinoma, NOS/) {
        system("cp Kidney_cancer_tumor_data/$a[1] papillary_adenocarcinoma_NOS");
    }
    elsif($sukhen =~ /Renal cell carcinoma, chromophobe type/) {
        system("cp Kidney_cancer_tumor_data/$a[1] renal_cell_carcinoma_chromophoba_type");
       }
    elsif($sukhen =~ /Wilms tumor/) {
        system("cp Kidney_cancer_tumor_data/$a[1] wilms_tumor");
    }
    else {
        system("cp Kidney_cancer_tumor_data/$a[1] not_matched");
    }
}

# Close the data file
close(DATA1);

