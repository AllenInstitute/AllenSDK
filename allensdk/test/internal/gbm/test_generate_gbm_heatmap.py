import pytest
import allensdk.internal.pipeline_modules.gbm.generate_gbm_heatmap as heatmap
import pandas as pd
import os
import numpy as np


TEST_DIR = os.path.dirname(__file__)
TEST_GENE_FILE = os.path.join(TEST_DIR, "test.genes.results")
TEST_TRANSCRIPT_FILE = os.path.join(TEST_DIR, "test.isoforms.results")
TEST2_GENE_FILE = os.path.join(TEST_DIR, "test2.genes.results")
TEST2_TRANSCRIPT_FILE = os.path.join(TEST_DIR, "test2.isoforms.results")


def test_create_transcripts_for_genes():

    analysis_run_gene_file = {"analysis_run_gene_path": TEST_GENE_FILE, "analysis_run_transcript_path":
                                                                        TEST_TRANSCRIPT_FILE, "rna_well_id": 300173630}

    data = heatmap.create_transcripts_for_genes(analysis_run_gene_file)
    d = [["gene_id", "transcript_id(s)"],
         ["1000", "NM_001792_3"],
         ["124989", "NM_001195192_1,NM_152347_4"],
         ["100008586", "NM_001098405_1"]]
    expected_data = pd.DataFrame(data=d)
    assert(expected_data.equals(data))


def test_create_genes_for_transcripts():

    analysis_run_transcript_file = {"analysis_run_gene_path": TEST_GENE_FILE, "analysis_run_transcript_path":
                                                                              TEST_TRANSCRIPT_FILE, "rna_well_id":
                                                                                                    300173630}

    data = heatmap.create_genes_for_transcripts(analysis_run_transcript_file)
    d = [["transcript_id", "gene_id"],
         ["NM_000015_2", "10"],
         ["NM_130786_3", "1"],
         ["tRNA-Tyr.100009601.chr14", "100"]]
    expected_data = pd.DataFrame(data=d)
    assert(expected_data.equals(data))


def test_create_gene_fpkm_table():

    analysis_run_records = [{"analysis_run_gene_path": TEST_GENE_FILE, "analysis_run_transcript_path":
                                                                       TEST_TRANSCRIPT_FILE, "rna_well_id": 300173630},
                            {"analysis_run_gene_path": TEST2_GENE_FILE, "analysis_run_transcript_path":
                                                                        TEST2_TRANSCRIPT_FILE, "rna_well_id": 300173634}]

    data = heatmap.create_gene_fpkm_table(analysis_run_records)
    d = np.column_stack([["108.14", "5.10", "0.00"], ["11.05", "21.41", "11.57"]])
    expected_data = pd.DataFrame(data=d, columns=[300173630, 300173634], index=[1000, 124989, 100008586])
    assert (expected_data.equals(data))


def test_create_transcript_fpkm_table():

    analysis_run_records = [{"analysis_run_gene_path": TEST_GENE_FILE, "analysis_run_transcript_path":
                                                                       TEST_TRANSCRIPT_FILE, "rna_well_id": 300173630},
                            {"analysis_run_gene_path": TEST2_GENE_FILE, "analysis_run_transcript_path":
                                                                        TEST2_TRANSCRIPT_FILE, "rna_well_id": 300173634}]

    data = heatmap.create_transcript_fpkm_table(analysis_run_records)
    d = np.column_stack([["10.00", "100.00", "100.00"], ["0.00", "100.00", "0.00"]])
    expected_data = pd.DataFrame(data=d, columns=[300173630, 300173634], index=["NM_000015_2", "NM_130786_3",
                                                                                "tRNA-Tyr.100009601.chr14"])
    assert(expected_data.equals(data))


def test_create_sample_metadata():

    sample_metadata_records = [
      {
          "structure_name": "Microvascular proliferation sampled by reference histology",
          "structure_id": 309780906,
          "structure_color": "ff330",
          "block_id": 703397,
          "polygon_id": 298726077,
          "specimen_id": 297710593,
          "tumor_name": "W1-1-2",
          "rna_well_id": 300173634,
          "structure_abbreviation": "CTmvp-reference-histology",
          "block_name": "W1-1-2-D.2",
          "tumor_id": 703393,
          "specimen_name": "W1-1-2-D.2.01"
      },
      {
        "structure_name": "Cellular Tumor sampled by reference histology",
        "structure_id": 309780592,
        "structure_color": "5d04",
        "block_id": 703397,
        "polygon_id": 298727153,
        "specimen_id": 297710593,
        "tumor_name": "W1-1-2",
        "rna_well_id": 300173630,
        "structure_abbreviation": "CT-reference-histology",
        "block_name": "W1-1-2-D.2",
        "tumor_id": 703393,
        "specimen_name": "W1-1-2-D.2.01"
      }
    ]
    data = heatmap.create_sample_metadata(sample_metadata_records)
    d = [[300173630, 703397, "W1-1-2-D.2", 298727153, 297710593, "W1-1-2-D.2.01", "CT-reference-histology", "5d04",
          309780592, "Cellular Tumor sampled by reference histology", 703393, "W1-1-2"],
         [300173634, 703397, "W1-1-2-D.2", 298726077, 297710593, "W1-1-2-D.2.01", "CTmvp-reference-histology", "ff330",
          309780906, "Microvascular proliferation sampled by reference histology", 703393, "W1-1-2"]]

    expected_data = pd.DataFrame(data=d, columns=["rna_well_id", "block_id", "block_name", "polygon_id", "specimen_id",
                                                  "specimen_name", "structure_abbreviation", "structure_color",
                                                  "structure_id", "structure_name", "tumor_id", "tumor_name"])
    pd.testing.assert_frame_equal(expected_data, data, check_like=True)
