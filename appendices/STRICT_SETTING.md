## Appendix: Selected Words in the Strict Setting

We select and refine the word list based on the following steps:

1. **Original list made from prefixes and top 200 highest-$F_1$ words from the top 200 highest-$F_1$ binaries:**

   ```
   repair, redraw, properties, priority, ocaml, rpl, exception, mh, translate, ya, off, ug, lk, cx, notifier, modem, fget, little, mutex, ob, ikrt, soap, pg, org, compress, sm, engine, parent, rb, 9, backup, scale, cc, clone, unpack, ex, diff, vfs, qapi, xdr, star, dispatch, ods, csu, av, pci, nd, vmfs, im, etc, initialize, pu, dict, visit, usal, sample
   ```

2. **Manually checked potential prefixes from the remaining top 200 highest-$F_1$ words:**

   ```
   mp, scsi, emulator, option, sc, curry
   ```

3. **The following words are incorrect prefixes and removed from the list as they are single words, or appear only once.**

   ```
   org, modem, cc, sample, rb, pu, ex, ug, star, nd, parent, ob, sm
   ```

4. **The following words are incorrect prefixes and removed from the list as they are normal words occurring in different projects with different function bodies with low $F_1$ scores and no duplicates:**

   ```
   priority, backup, scale, translate, im, repair, dispatch, compress, option, redraw
   ```

5. **Final word list:**

   ```
   dict, lk, emulator, vfs, ya, off, etc, initialize, pg, vmfs, diff, csu, cx, visit, xdr, av, pci, usal, mh, mutex, unpack, sc, rpl, ocaml, properties, notifier, fget, engine, 9, little, scsi, translate, ods, soap, mp, exception, clone, qapi, ikrt, curry
   ```

Most duplicates with prefix words are from statically-linked runtime functions or library functions.  
Table 1 includes some cases where duplicate functions are found across different packages in training set and test set.  
Besides, the following words are prefix without duplicates, and we keep them in the word list:

```
mh, lk, etc, ikrt, rpl, vmfs, sc, engine, vfs
```

| **Test Set Package** | **Training Set Package** | **Corresponding Words**                                  |
|----------------------|--------------------------|-----------------------------------------------------------|
| pgbouncer            | pgqd                     | cx, pg                                                    |
| libhsm-bin           | opendnssec-enforcer-sqlite3 | ods                                                     |
| wmmatrix             | xscreensaver-data-extra  | ya                                                        |
| knot-dnsutils        | knot                     | mp                                                        |
| qemu-ga              | qemu-utils               | properties, mutex off, clone notifier, visit qapi, dict emulator, pci |
| *.cmxs              | *.cmxs                   | ocaml, curry, 9                                           |
| gridsite-clients     | lfc                      | soap, exception                                           |
| quota                | rstatd                   | xdr, diff                                                 |
| genisoimage          | wodim                    | usal, scsi, av, fget                                      |

**Table 1:** Packages sharing similar functionalities and their corresponding words.

