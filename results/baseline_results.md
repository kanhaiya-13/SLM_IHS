# Baseline Results

| model | state | level | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | Maharashtra | window | 0.5429 | 0.7170 | 0.6179 | 0.3879 | 0.5616 |
| persistence | Maharashtra | t+4 | 0.3191 | 0.3846 | 0.3488 | 0.3590 | 0.3986 |
| persistence | Maharashtra | t+5 | 0.3043 | 0.3684 | 0.3333 | 0.3577 | 0.3880 |
| persistence | Maharashtra | t+6 | 0.3043 | 0.3684 | 0.3333 | 0.3577 | 0.3880 |
| lstm | Maharashtra | window | 0.6047 | 0.9811 | 0.7482 | 0.5988 | 0.7650 |
| lstm | Maharashtra | t+4 | 0.4483 | 1.0000 | 0.6190 | 0.5454 | 0.4945 |
| lstm | Maharashtra | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.2755 | 0.3311 |
| lstm | Maharashtra | t+6 | 0.4368 | 1.0000 | 0.6080 | 0.6386 | 0.6853 |
| persistence | Karnataka | window | 0.0857 | 0.1667 | 0.1132 | 0.3514 | 0.1867 |
| persistence | Karnataka | t+4 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| persistence | Karnataka | t+5 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| persistence | Karnataka | t+6 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| lstm | Karnataka | window | 0.0000 | 0.0000 | 0.0000 | 0.5926 | 0.3158 |
| lstm | Karnataka | t+4 | 0.0920 | 1.0000 | 0.1684 | 0.3623 | 0.0766 |
| lstm | Karnataka | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.3038 | 0.0712 |
| lstm | Karnataka | t+6 | 0.0920 | 1.0000 | 0.1684 | 0.2532 | 0.0664 |
| persistence | Tamil Nadu | window | 0.1538 | 0.2105 | 0.1778 | 0.4435 | 0.2048 |
| persistence | Tamil Nadu | t+4 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| persistence | Tamil Nadu | t+5 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| persistence | Tamil Nadu | t+6 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| lstm | Tamil Nadu | window | 0.2405 | 1.0000 | 0.3878 | 0.4226 | 0.1852 |
| lstm | Tamil Nadu | t+4 | 0.0920 | 1.0000 | 0.1684 | 0.2864 | 0.0686 |
| lstm | Tamil Nadu | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.4826 | 0.0937 |
| lstm | Tamil Nadu | t+6 | 0.0920 | 1.0000 | 0.1684 | 0.4684 | 0.0910 |
