# Ablation Table

| model | state | CI | weather | calendar | level | precision | recall | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer (CI_W_C) | Maharashtra | 1.0000 | 1.0000 | 1.0000 | window | 0.5976 | 0.9245 | 0.7259 | 0.6099 | 0.7796 |
| Transformer (CI_W_C) | Maharashtra | 1.0000 | 1.0000 | 1.0000 | t+4 | 0.6333 | 0.4872 | 0.5507 | 0.7099 | 0.6169 |
| Transformer (CI_W_C) | Maharashtra | 1.0000 | 1.0000 | 1.0000 | t+5 | 0.5349 | 0.6053 | 0.5679 | 0.5892 | 0.5008 |
| Transformer (CI_W_C) | Maharashtra | 1.0000 | 1.0000 | 1.0000 | t+6 | 0.5250 | 0.5526 | 0.5385 | 0.5644 | 0.4906 |
| Transformer (CM_W_C) | Maharashtra | 0.0000 | 1.0000 | 1.0000 | window | 0.6889 | 0.5849 | 0.6327 | 0.6582 | 0.7920 |
| Transformer (CM_W_C) | Maharashtra | 0.0000 | 1.0000 | 1.0000 | t+4 | 0.8235 | 0.3590 | 0.5000 | 0.7623 | 0.7452 |
| Transformer (CM_W_C) | Maharashtra | 0.0000 | 1.0000 | 1.0000 | t+5 | 0.8824 | 0.3947 | 0.5455 | 0.7669 | 0.7509 |
| Transformer (CM_W_C) | Maharashtra | 0.0000 | 1.0000 | 1.0000 | t+6 | 0.6957 | 0.4211 | 0.5246 | 0.7589 | 0.7288 |
| Transformer (CI_W_NC) | Maharashtra | 1.0000 | 1.0000 | 0.0000 | window | 0.7164 | 0.9057 | 0.8000 | 0.7786 | 0.8466 |
| Transformer (CI_W_NC) | Maharashtra | 1.0000 | 1.0000 | 0.0000 | t+4 | 0.7391 | 0.4359 | 0.5484 | 0.7660 | 0.7535 |
| Transformer (CI_W_NC) | Maharashtra | 1.0000 | 1.0000 | 0.0000 | t+5 | 0.5862 | 0.8947 | 0.7083 | 0.7121 | 0.5966 |
| Transformer (CI_W_NC) | Maharashtra | 1.0000 | 1.0000 | 0.0000 | t+6 | 0.5625 | 0.7105 | 0.6279 | 0.6224 | 0.4875 |
| Transformer (CI_NW_C) | Maharashtra | 1.0000 | 0.0000 | 1.0000 | window | 0.6333 | 0.7170 | 0.6726 | 0.6154 | 0.7682 |
| Transformer (CI_NW_C) | Maharashtra | 1.0000 | 0.0000 | 1.0000 | t+4 | 0.5806 | 0.4615 | 0.5143 | 0.6506 | 0.6443 |
| Transformer (CI_NW_C) | Maharashtra | 1.0000 | 0.0000 | 1.0000 | t+5 | 0.5667 | 0.4474 | 0.5000 | 0.6831 | 0.5665 |
| Transformer (CI_NW_C) | Maharashtra | 1.0000 | 0.0000 | 1.0000 | t+6 | 0.5128 | 0.5263 | 0.5195 | 0.6552 | 0.6143 |
| Transformer (CI_NW_NC) | Maharashtra | 1.0000 | 0.0000 | 0.0000 | window | 0.7143 | 0.8491 | 0.7759 | 0.5888 | 0.6162 |
| Transformer (CI_NW_NC) | Maharashtra | 1.0000 | 0.0000 | 0.0000 | t+4 | 0.5758 | 0.4872 | 0.5278 | 0.6442 | 0.5416 |
| Transformer (CI_NW_NC) | Maharashtra | 1.0000 | 0.0000 | 0.0000 | t+5 | 0.5000 | 0.4211 | 0.4571 | 0.6144 | 0.4806 |
| Transformer (CI_NW_NC) | Maharashtra | 1.0000 | 0.0000 | 0.0000 | t+6 | 0.4194 | 0.3421 | 0.3768 | 0.5650 | 0.4402 |
| Transformer (CI_W_C) | Karnataka | 1.0000 | 1.0000 | 1.0000 | window | 0.0588 | 0.0556 | 0.0571 | 0.5089 | 0.2015 |
| Transformer (CI_W_C) | Karnataka | 1.0000 | 1.0000 | 1.0000 | t+4 | 0.0833 | 0.1250 | 0.1000 | 0.7199 | 0.1636 |
| Transformer (CI_W_C) | Karnataka | 1.0000 | 1.0000 | 1.0000 | t+5 | 0.1818 | 0.2500 | 0.2105 | 0.6361 | 0.1850 |
| Transformer (CI_W_C) | Karnataka | 1.0000 | 1.0000 | 1.0000 | t+6 | 0.0714 | 0.1250 | 0.0909 | 0.7278 | 0.1649 |
| Transformer (CM_W_C) | Karnataka | 0.0000 | 1.0000 | 1.0000 | window | 0.1154 | 0.3333 | 0.1714 | 0.2295 | 0.1395 |
| Transformer (CM_W_C) | Karnataka | 0.0000 | 1.0000 | 1.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.3877 | 0.0821 |
| Transformer (CM_W_C) | Karnataka | 0.0000 | 1.0000 | 1.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.2943 | 0.0702 |
| Transformer (CM_W_C) | Karnataka | 0.0000 | 1.0000 | 1.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.2975 | 0.0694 |
| Transformer (CI_W_NC) | Karnataka | 1.0000 | 1.0000 | 0.0000 | window | 0.1250 | 0.3333 | 0.1818 | 0.3816 | 0.1850 |
| Transformer (CI_W_NC) | Karnataka | 1.0000 | 1.0000 | 0.0000 | t+4 | 0.2500 | 0.5000 | 0.3333 | 0.7540 | 0.2751 |
| Transformer (CI_W_NC) | Karnataka | 1.0000 | 1.0000 | 0.0000 | t+5 | 1.0000 | 0.2500 | 0.4000 | 0.8093 | 0.5203 |
| Transformer (CI_W_NC) | Karnataka | 1.0000 | 1.0000 | 0.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.6946 | 0.1506 |
| Transformer (CI_NW_C) | Karnataka | 1.0000 | 0.0000 | 1.0000 | window | 0.2000 | 0.4444 | 0.2759 | 0.3800 | 0.1848 |
| Transformer (CI_NW_C) | Karnataka | 1.0000 | 0.0000 | 1.0000 | t+4 | 0.2941 | 0.6250 | 0.4000 | 0.7223 | 0.2356 |
| Transformer (CI_NW_C) | Karnataka | 1.0000 | 0.0000 | 1.0000 | t+5 | 0.1667 | 0.1250 | 0.1429 | 0.5657 | 0.1426 |
| Transformer (CI_NW_C) | Karnataka | 1.0000 | 0.0000 | 1.0000 | t+6 | 0.4000 | 0.2500 | 0.3077 | 0.6693 | 0.2483 |
| Transformer (CI_NW_NC) | Karnataka | 1.0000 | 0.0000 | 0.0000 | window | 0.0476 | 0.0556 | 0.0513 | 0.3341 | 0.1662 |
| Transformer (CI_NW_NC) | Karnataka | 1.0000 | 0.0000 | 0.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.3845 | 0.0778 |
| Transformer (CI_NW_NC) | Karnataka | 1.0000 | 0.0000 | 0.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.4446 | 0.0851 |
| Transformer (CI_NW_NC) | Karnataka | 1.0000 | 0.0000 | 0.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.6036 | 0.1177 |
| Transformer (CI_W_C) | Tamil Nadu | 1.0000 | 1.0000 | 1.0000 | window | 0.1875 | 0.4737 | 0.2687 | 0.4853 | 0.2352 |
| Transformer (CI_W_C) | Tamil Nadu | 1.0000 | 1.0000 | 1.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.3940 | 0.0803 |
| Transformer (CI_W_C) | Tamil Nadu | 1.0000 | 1.0000 | 1.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.3354 | 0.0730 |
| Transformer (CI_W_C) | Tamil Nadu | 1.0000 | 1.0000 | 1.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.2516 | 0.0655 |
| Transformer (CM_W_C) | Tamil Nadu | 0.0000 | 1.0000 | 1.0000 | window | 0.2951 | 0.9474 | 0.4500 | 0.7337 | 0.4750 |
| Transformer (CM_W_C) | Tamil Nadu | 0.0000 | 1.0000 | 1.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.6187 | 0.1434 |
| Transformer (CM_W_C) | Tamil Nadu | 0.0000 | 1.0000 | 1.0000 | t+5 | 0.2857 | 0.5000 | 0.3636 | 0.7231 | 0.2504 |
| Transformer (CM_W_C) | Tamil Nadu | 0.0000 | 1.0000 | 1.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.6883 | 0.1762 |
| Transformer (CI_W_NC) | Tamil Nadu | 1.0000 | 1.0000 | 0.0000 | window | 0.1458 | 0.3684 | 0.2090 | 0.3417 | 0.1759 |
| Transformer (CI_W_NC) | Tamil Nadu | 1.0000 | 1.0000 | 0.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.3252 | 0.0758 |
| Transformer (CI_W_NC) | Tamil Nadu | 1.0000 | 1.0000 | 0.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.3093 | 0.0741 |
| Transformer (CI_W_NC) | Tamil Nadu | 1.0000 | 1.0000 | 0.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.3236 | 0.0718 |
| Transformer (CI_NW_C) | Tamil Nadu | 1.0000 | 0.0000 | 1.0000 | window | 0.2000 | 0.5789 | 0.2973 | 0.4265 | 0.2563 |
| Transformer (CI_NW_C) | Tamil Nadu | 1.0000 | 0.0000 | 1.0000 | t+4 | 0.0435 | 0.1250 | 0.0645 | 0.4462 | 0.0890 |
| Transformer (CI_NW_C) | Tamil Nadu | 1.0000 | 0.0000 | 1.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.5158 | 0.1230 |
| Transformer (CI_NW_C) | Tamil Nadu | 1.0000 | 0.0000 | 1.0000 | t+6 | 0.1250 | 0.1250 | 0.1250 | 0.4272 | 0.2132 |
| Transformer (CI_NW_NC) | Tamil Nadu | 1.0000 | 0.0000 | 0.0000 | window | 0.1579 | 0.1579 | 0.1579 | 0.4539 | 0.2077 |
| Transformer (CI_NW_NC) | Tamil Nadu | 1.0000 | 0.0000 | 0.0000 | t+4 | 0.0000 | 0.0000 | 0.0000 | 0.6653 | 0.1612 |
| Transformer (CI_NW_NC) | Tamil Nadu | 1.0000 | 0.0000 | 0.0000 | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.5182 | 0.1146 |
| Transformer (CI_NW_NC) | Tamil Nadu | 1.0000 | 0.0000 | 0.0000 | t+6 | 0.0000 | 0.0000 | 0.0000 | 0.4850 | 0.0911 |
| persistence | Maharashtra | - | - | - | window | 0.5429 | 0.7170 | 0.6179 | 0.3879 | 0.5616 |
| persistence | Maharashtra | - | - | - | t+4 | 0.3191 | 0.3846 | 0.3488 | 0.3590 | 0.3986 |
| persistence | Maharashtra | - | - | - | t+5 | 0.3043 | 0.3684 | 0.3333 | 0.3577 | 0.3880 |
| persistence | Maharashtra | - | - | - | t+6 | 0.3043 | 0.3684 | 0.3333 | 0.3577 | 0.3880 |
| lstm | Maharashtra | - | - | - | window | 0.6047 | 0.9811 | 0.7482 | 0.5988 | 0.7650 |
| lstm | Maharashtra | - | - | - | t+4 | 0.4483 | 1.0000 | 0.6190 | 0.5454 | 0.4945 |
| lstm | Maharashtra | - | - | - | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.2755 | 0.3311 |
| lstm | Maharashtra | - | - | - | t+6 | 0.4368 | 1.0000 | 0.6080 | 0.6386 | 0.6853 |
| persistence | Karnataka | - | - | - | window | 0.0857 | 0.1667 | 0.1132 | 0.3514 | 0.1867 |
| persistence | Karnataka | - | - | - | t+4 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| persistence | Karnataka | - | - | - | t+5 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| persistence | Karnataka | - | - | - | t+6 | 0.0667 | 0.1250 | 0.0870 | 0.4739 | 0.0888 |
| lstm | Karnataka | - | - | - | window | 0.0000 | 0.0000 | 0.0000 | 0.5926 | 0.3158 |
| lstm | Karnataka | - | - | - | t+4 | 0.0920 | 1.0000 | 0.1684 | 0.3623 | 0.0766 |
| lstm | Karnataka | - | - | - | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.3038 | 0.0712 |
| lstm | Karnataka | - | - | - | t+6 | 0.0920 | 1.0000 | 0.1684 | 0.2532 | 0.0664 |
| persistence | Tamil Nadu | - | - | - | window | 0.1538 | 0.2105 | 0.1778 | 0.4435 | 0.2048 |
| persistence | Tamil Nadu | - | - | - | t+4 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| persistence | Tamil Nadu | - | - | - | t+5 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| persistence | Tamil Nadu | - | - | - | t+6 | 0.0909 | 0.1250 | 0.1053 | 0.4992 | 0.0918 |
| lstm | Tamil Nadu | - | - | - | window | 0.2405 | 1.0000 | 0.3878 | 0.4226 | 0.1852 |
| lstm | Tamil Nadu | - | - | - | t+4 | 0.0920 | 1.0000 | 0.1684 | 0.2864 | 0.0686 |
| lstm | Tamil Nadu | - | - | - | t+5 | 0.0000 | 0.0000 | 0.0000 | 0.4826 | 0.0937 |
| lstm | Tamil Nadu | - | - | - | t+6 | 0.0920 | 1.0000 | 0.1684 | 0.4684 | 0.0910 |
