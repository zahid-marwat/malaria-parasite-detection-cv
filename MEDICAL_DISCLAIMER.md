# Medical Disclaimer

## IMPORTANT: RESEARCH AND EDUCATIONAL USE ONLY

**This system is NOT approved for clinical diagnostic use and should NOT be used to make medical decisions without expert human review.**

### Intended Use
This malaria parasite detection system is developed for:
- **Research purposes only**
- **Educational and training purposes**
- **Proof-of-concept demonstrations**
- **Preliminary screening only** (with mandatory expert validation)

### Clinical Limitations
1. **Not a substitute for medical professionals**: This system cannot replace microscopy experts, pathologists, or clinicians in making diagnostic decisions.

2. **Regulatory status**: This system has NOT been validated for clinical use and does NOT comply with FDA 510(k) or other regulatory requirements for clinical diagnostic devices.

3. **Accuracy limitations**:
   - Performance varies based on image quality, staining variations, and dataset characteristics
   - The system may fail on:
     - Poor quality blood smear images
     - Non-standard staining protocols
     - Different microscope types
     - Mixed or atypical parasite presentations
   - False negatives (missed infections) and false positives (false alarms) are possible

4. **Data limitations**:
   - This system was trained on the ThickBloodSmears_150 dataset
   - Generalization to other datasets or clinical settings is not guaranteed
   - Imbalanced training data may affect performance

5. **Confidence scores should not be interpreted as clinical probabilities**

### Mandatory Human Review
Any deployment of this system in clinical settings requires:
- ✅ Explicit validation by qualified parasitologists or pathologists
- ✅ Compliance with local medical regulations and standards
- ✅ Ethics board approval for any clinical trials
- ✅ Informed consent from study participants
- ✅ Insurance and liability coverage
- ✅ Audit trails for all predictions
- ✅ Clear communication to users that this is a screening tool only

### Clinical Best Practices
When using this system:
1. **Always validate predictions** with qualified microscopy experts
2. **Never make treatment decisions** based solely on this system's output
3. **Report failures and edge cases** to improve the system
4. **Prioritize sensitivity over specificity** (it's better to flag uncertain cases for expert review)
5. **Maintain complete audit logs** of all predictions
6. **Provide uncertainty estimates** with all predictions

### Ethical Considerations
- This technology should be used to **support**, not **replace**, human experts
- Ensure equitable access and avoid perpetuating healthcare disparities
- Be transparent about AI involvement in clinical decisions
- Protect patient privacy and data security
- Consider the limitations in resource-limited settings

### Liability
The developers and contributors of this system assume NO liability for:
- Misdiagnosis resulting from use of this system
- Clinical decisions made based on this system's predictions
- Adverse patient outcomes
- Regulatory violations
- Data breaches or privacy violations

Users must assume full responsibility for proper clinical validation and regulatory compliance before any clinical deployment.

### Risk Stratification
**Risk Levels**:
- 🔴 **HIGH RISK**: Using this system as the sole diagnostic method
- 🟡 **MEDIUM RISK**: Using for preliminary screening with clinical validation
- 🟢 **LOW RISK**: Using for research, education, or development purposes

This system is appropriate only for GREEN and YELLOW contexts with proper safeguards.

### Contact and Reporting
For issues, improvements, or ethical concerns, please contact the development team through the project repository.

---

**Last Updated**: February 2026
**Version**: 1.0
**Status**: Research/Educational Use Only
