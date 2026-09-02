"""Golden dataset for Employee Handbook RAG retrieval evaluation."""

from deepeval.dataset import Golden

rag_retrieval_goldens = [
    # ============================================================
    # EMPLOYMENT BASICS
    # ============================================================
    Golden(
        input="How many hours per week must a full-time employee work?",
        expected_output="At least 30 hours per week on average (or 130 hours per month on average).",
    ),
    Golden(
        input="What happens if I'm a non-exempt employee who works overtime?",
        expected_output="Non-exempt employees are entitled to overtime pay at one and a half times their wage under the handbook's U.S. overtime provision. Overtime hours should be recorded accurately and worked only after supervisor authorization.",
    ),
    Golden(
        input="What types of contracts can full-time and part-time employees have?",
        expected_output="Both full-time and part-time employees can have either temporary or indefinite-duration contracts.",
    ),
    Golden(
        input="Are full-time employees with indefinite contracts entitled to benefits?",
        expected_output="Yes. Full-time employees under an indefinite-duration contract are entitled to the company's full benefits package.",
    ),
    # ============================================================
    # EQUAL OPPORTUNITY
    # ============================================================
    Golden(
        input="What is the company's policy on discrimination?",
        expected_output="The company is an equal opportunity employer and does not tolerate discrimination against protected characteristics. Employees are expected to treat others with respect and professionalism, and discriminatory behavior may result in disciplinary action.",
    ),
    Golden(
        input="Which characteristics are protected under the equal opportunity policy?",
        expected_output="The handbook lists gender, age, sexual orientation, race, nationality, ethnicity, religion, disability, and veteran status as protected characteristics.",
    ),
    Golden(
        input="What should an employee do if they experience discrimination?",
        expected_output="Report the discriminatory action to HR, whether it affects the employee or a colleague.",
    ),
    Golden(
        input="Can an employee be retaliated against for filing a discrimination complaint?",
        expected_output="No. The company states that it will not retaliate against an employee for filing a complaint or discrimination lawsuit. Retaliation may result in disciplinary action.",
    ),
    # ============================================================
    # RECRUITMENT
    # ============================================================
    Golden(
        input="What are the main steps in the company's recruitment process?",
        expected_output="The process is: identify the need for a job opening; decide whether to hire externally or internally; review the job description and write the ad; get approval; select posting sources; decide hiring stages and timeframes; review resumes in the company database/ATS; source passive candidates; shortlist applicants; screen and interview; run background checks and check references; select the most suitable candidate; and make an official offer. Steps may overlap or be skipped when appropriate.",
    ),
    Golden(
        input="When should background checks generally be conducted?",
        expected_output="As a general rule, background checks should be commissioned for finalists only.",
    ),
    Golden(
        input="What is required before conducting a candidate background check?",
        expected_output="Ask HR for guidance, use the contracted provider, comply with applicable laws, ensure the candidate understands the process, and obtain the candidate's permission.",
    ),
    # ============================================================
    # EMPLOYEE REFERRALS
    # ============================================================
    Golden(
        input="How much is the standard employee referral bonus?",
        expected_output="The standard referral reward is a $3,000 referral bonus (or the alternative reward specified in the handbook).",
    ),
    Golden(
        input="How much can an employee receive for referring someone to a hard-to-fill role?",
        expected_output="The reward may be higher for a hard-to-fill role; the handbook gives $6,000 for a referred Data Scientist as an example.",
    ),
    Golden(
        input="Is there a limit on how many employee referrals someone can make?",
        expected_output="No. There is no cap on the number of referrals an employee can make, and eligible rewards are paid accordingly.",
    ),
    Golden(
        input="What happens if two employees refer the same candidate?",
        expected_output="Only the first referrer receives the referral incentive.",
    ),
    Golden(
        input="What conditions must a referred candidate meet?",
        expected_output="The candidate must not have applied to the company for at least a year and must be hired as a permanent full- or part-time employee, not as a temporary employee or contractor.",
    ),
    # ============================================================
    # ATTENDANCE
    # ============================================================
    Golden(
        input="What should an employee do if an emergency prevents them from coming to work?",
        expected_output="Contact their manager as soon as possible. The handbook says unreported absences may be excused in cases such as serious accidents or acute medical emergencies.",
    ),
    Golden(
        input="Does the company expect employees to attend their scheduled working hours?",
        expected_output="Yes. Employees are expected to be present during their scheduled working hours.",
    ),
    # ============================================================
    # CONFIDENTIALITY AND DATA PROTECTION
    # ============================================================
    Golden(
        input="What types of information are considered confidential?",
        expected_output="Examples include employee records, unpublished financial information, customer/partner/vendor data, existing and prospective customer lists, and unpublished goals, forecasts, and initiatives marked as confidential.",
    ),
    Golden(
        input="How should employees protect confidential information?",
        expected_output="Lock or secure it at all times; shred confidential documents when no longer needed; view it only on secure devices; disclose it internally only when necessary and authorized; and keep confidential documents on company premises unless moving them is absolutely necessary.",
    ),
    Golden(
        input="Can employees disclose confidential company information to people outside the company?",
        expected_output="No. Employees must not disclose confidential information to anyone outside the company.",
    ),
    Golden(
        input="Can employees use confidential company information for personal profit?",
        expected_output="No. Employees must not use confidential information for personal benefit or profit.",
    ),
    Golden(
        input="What can happen if an employee breaches confidentiality rules for personal profit?",
        expected_output="The company states that it will terminate an employee who breaches confidentiality guidelines for personal profit.",
    ),
    # ============================================================
    # HARASSMENT
    # ============================================================
    Golden(
        input="What behaviors can be considered workplace harassment?",
        expected_output="Examples include intentionally sabotaging someone's work, frequent or unwanted advances, derogatory comments about ethnic heritage or religious beliefs, spreading rumors about someone's personal life, and ridiculing or singling someone out for unwanted tasks. The handbook notes that harassment is broad and can include seemingly harmless actions such as gossip.",
    ),
    Golden(
        input="What should an employee do if they are experiencing workplace harassment?",
        expected_output="They can speak to the offender in minor cases when appropriate, contact their manager, or contact HR. HR can be contacted for any harassment, and serious harassment or a claim involving the manager should be reported to HR as soon as possible.",
    ),
    Golden(
        input="Who can an employee report harassment to?",
        expected_output="An employee can report harassment to the offender in appropriate minor cases, their manager, or HR. HR can be contacted in any case of harassment.",
    ),
    Golden(
        input="What happens when an employee is found guilty of sexual harassment?",
        expected_output="The handbook states that the employee will be terminated.",
    ),
    # ============================================================
    # WORKPLACE VIOLENCE
    # ============================================================
    Golden(
        input="What does the company consider workplace violence?",
        expected_output="Workplace violence includes physical and sexual assault, destruction of property, threats to harm a person or property, and verbal and psychological abuse.",
    ),
    Golden(
        input="What should an employee do if they suspect someone is being violent?",
        expected_output="Report the situation to HR. The report will be treated confidentially and investigated with discretion.",
    ),
    Golden(
        input="What should an employee do if they witness severe physical violence?",
        expected_output="Call the building's security and avoid getting involved for their own safety, particularly in incidents involving a lethal weapon.",
    ),
    # ============================================================
    # WORKPLACE SAFETY
    # ============================================================
    Golden(
        input="What measures does the company take to prevent workplace injuries?",
        expected_output="The company periodically conducts risk assessments and job hazard analyses and establishes preventative measures. It may provide safety training, protect employees in dangerous locations, provide protective gear, and have inspectors/quality-control employees regularly evaluate equipment and infrastructure.",
    ),
    Golden(
        input="What emergency management measures are provided?",
        expected_output="The handbook lists regularly inspected smoke alarms and sprinklers, technicians available to address leakages, damages, and blackouts, accessible fire extinguishers and other fire-protection equipment, evacuation plans posted on each floor and online, and clearly indicated fire escapes and safety exits.",
    ),
    # ============================================================
    # SMOKING AND DRUG-FREE WORKPLACE
    # ============================================================
    Golden(
        input="Is the workplace smoke-free?",
        expected_output="Yes. The company is a smoke-free workplace. Smoking is allowed only in the designated areas specified by the company; other workplace areas are strictly smoke-free.",
    ),
    Golden(
        input="Are drugs allowed on company premises?",
        expected_output="No. Employees, contractors, and visitors must not bring, use, give away, or sell drugs on company premises.",
    ),
    Golden(
        input="What can happen if an employee is caught with illegal drugs at work?",
        expected_output="They may face disciplinary action up to and including termination.",
    ),
    # ============================================================
    # INTERNET AND DIGITAL DEVICES
    # ============================================================
    Golden(
        input="Can employees use the company's internet for personal purposes?",
        expected_output="Yes, occasionally, as long as personal use does not interfere with job responsibilities. Employees may also be asked to stop personal activities that slow the corporate connection.",
    ),
    Golden(
        input="What activities are prohibited when using the company's internet?",
        expected_output="Employees must not use it to download/upload obscene, offensive, or illegal material; send confidential information to unauthorized recipients; invade privacy or access sensitive information; pirate media/software; visit potentially dangerous sites that could compromise security; or perform unauthorized/illegal actions such as hacking, fraud, or buying/selling illegal goods.",
    ),
    Golden(
        input="Can employees use their phones while driving a company vehicle?",
        expected_output="No. Employees should not use their phones for any reason while driving a company vehicle.",
    ),
    Golden(
        input="Can employees use their phones to record confidential information?",
        expected_output="No. Employees must not use their phone to record confidential information.",
    ),
    # ============================================================
    # CORPORATE EMAIL
    # ============================================================
    Golden(
        input="Can employees use corporate email for personal purposes?",
        expected_output="Yes. Limited personal use is allowed if employees keep the email safe and avoid spamming or disclosing confidential information.",
    ),
    Golden(
        input="What should employees avoid doing with corporate email?",
        expected_output="They should avoid signing up for illegal, unreliable, disreputable, or suspect services; sending unauthorized marketing; registering for competitors' services unless authorized; sending insulting or discriminatory content; and intentionally spamming others. They should also use strong passwords and watch for malware and phishing.",
    ),
    Golden(
        input="What should employees do if they are unsure whether an email is safe?",
        expected_output="Ask the company's Security Specialists.",
    ),
    # ============================================================
    # SOCIAL MEDIA
    # ============================================================
    Golden(
        input="Can employees access personal social media accounts at work?",
        expected_output="Yes, but they must use them responsibly and remain productive.",
    ),
    Golden(
        input="Can employees share confidential company information on social media?",
        expected_output="No. Employees should not share confidential information or intellectual property on social media.",
    ),
    Golden(
        input="What should an employee do before sharing unannounced company news?",
        expected_output="Ask their manager or PR first.",
    ),
    # ============================================================
    # CONFLICT OF INTEREST
    # ============================================================
    Golden(
        input="What is a conflict of interest according to the handbook?",
        expected_output="It is a situation in which an employee's personal goals are no longer aligned with their responsibilities to the company. The handbook gives accepting a bribe for personal financial benefit as an example.",
    ),
    Golden(
        input="What should an employee do if they face an ethical dilemma?",
        expected_output="Talk to their manager or HR so they can help resolve the dilemma.",
    ),
    # ============================================================
    # EMPLOYEE RELATIONSHIPS
    # ============================================================
    Golden(
        input="Can employees date their coworkers?",
        expected_output="Yes, consensual dating between colleagues is permitted, but employees must remain professional, keep personal discussions outside the workplace, and respect colleagues who date.",
    ),
    Golden(
        input="Can a manager date their direct report?",
        expected_output="No. Supervisors must not date their direct reports, and the restriction extends to every manager above an employee.",
    ),
    Golden(
        input="Can a hiring manager hire their romantic partner onto their team?",
        expected_output="No. A hiring manager cannot hire their partner onto their team. They may refer the partner to another team or department where they have no managerial or hiring authority.",
    ),
    # ============================================================
    # EMPLOYMENT OF RELATIVES
    # ============================================================
    Golden(
        input="How does the company define a relative?",
        expected_output="A relative is someone related by blood or marriage within the third degree. The handbook includes parents, grandparents, in-laws, spouses/domestic partners, children, grandchildren, siblings, uncles, aunts, nieces, nephews, step-parents, step-children, and adopted children.",
    ),
    Golden(
        input="Can an employee supervise a relative?",
        expected_output="No. Employees must not be involved in a supervisory/reporting relationship with a relative.",
    ),
    Golden(
        input="Can an employee participate in a hiring committee when their relative is being interviewed?",
        expected_output="No. An employee cannot be part of a hiring committee when their relative is interviewed for that position.",
    ),
    # ============================================================
    # WORKPLACE VISITORS
    # ============================================================
    Golden(
        input="What should an employee do before bringing a visitor to the office?",
        expected_output="Ask permission from the designated HR Manager, Security Officer, or Office Manager and inform the designated reception/gate/front office of the visitor's arrival.",
    ),
    Golden(
        input="What must office visitors do when they arrive?",
        expected_output="Visitors should sign in, show identification, receive a pass, and return the pass when the visit is complete.",
    ),
    # ============================================================
    # SOLICITATION
    # ============================================================
    Golden(
        input="Does the company allow solicitation by non-employees?",
        expected_output="No. The company does not allow solicitation and distribution by non-employees in the workplace.",
    ),
    Golden(
        input="When may an employee solicit colleagues?",
        expected_output="Employees may solicit colleagues for certain authorized purposes: organizing events for another employee, supporting a company-sponsored/funded/organized/authorized cause or charity, inviting colleagues to authorized non-business activities, or participating in employment-related activities/groups protected by law. They must not disturb or distract colleagues from work.",
    ),
    # ============================================================
    # PERFORMANCE MANAGEMENT
    # ============================================================
    Golden(
        input="What are the goals of the company's performance management process?",
        expected_output="The goals are to ensure employees understand responsibilities and specific goals, provide actionable and timely feedback, invest in development, and recognize/reward work financially or non-financially.",
    ),
    Golden(
        input="Are pay increases or bonuses guaranteed after performance reviews?",
        expected_output="No. Pay increases and bonuses are not guaranteed, although managers are encouraged to recommend rewards when deserved.",
    ),
    Golden(
        input="How often should managers meet with their team members for feedback?",
        expected_output="Managers are instructed to meet with team members once per week.",
    ),
    # ============================================================
    # TRAINING AND DEVELOPMENT
    # ============================================================
    Golden(
        input="How much does each employee have annually for educational activities?",
        expected_output="Each employee has $1,000 annually for educational activities or materials. Subscriptions and books are included unless necessary for everyday duties.",
    ),
    Golden(
        input="What training opportunities does the company offer?",
        expected_output="Formal training sessions, employee coaching and mentoring, industry conferences, on-the-job training, job shadowing, and job rotation.",
    ),
    # ============================================================
    # WORK FROM HOME
    # ============================================================
    Golden(
        input="How often can an eligible employee normally work from home?",
        expected_output="The handbook normally allows one day of work from home per week. More days require discussion with the manager.",
    ),
    Golden(
        input="How far in advance should an employee request to work from home?",
        expected_output="The employee should inform their manager at least two days in advance, using the company's HRIS as specified in the handbook.",
    ),
    Golden(
        input="Can an employee work from home during a rare emergency without prior approval?",
        expected_output="Yes. In a rare emergency, an employee may work from home without prior approval, but should call or email their manager as soon as possible (or contact HR if the manager is in a different time zone).",
    ),
    Golden(
        input="What should employees consider when working from home?",
        expected_output="Use a fast and secure internet connection and secure devices, choose a place without loud noises or distractions, and check in with the team frequently to support collaboration.",
    ),
    # ============================================================
    # REMOTE WORKING
    # ============================================================
    Golden(
        input="How long can an office-based employee work remotely?",
        expected_output="An office-based employee may work remotely for a maximum of two consecutive weeks per year, subject to the handbook's conditions.",
    ),
    Golden(
        input="What reasons may allow an office-based employee to work remotely?",
        expected_output="The handbook specifically gives being a new parent or having a short-term disability as reasons; employees with another reason should talk to their manager.",
    ),
    Golden(
        input="How far in advance must remote working requests be submitted?",
        expected_output="Remote-working requests should be submitted at least one week in advance through the HRIS, as specified in the handbook.",
    ),
    Golden(
        input="What policies must permanent remote employees follow?",
        expected_output="They must follow the same security, confidentiality, and equal opportunity policies as their office-based colleagues.",
    ),
    # ============================================================
    # EMPLOYEE EXPENSES
    # ============================================================
    Golden(
        input="Which employee expenses can be reimbursed?",
        expected_output="Reimbursable categories include business travel, relocation, education and training, and, upon approval, outings with business partners or colleagues. The handbook notes that not all travel expenses are reimbursable.",
    ),
    Golden(
        input="How long do employees have to submit reimbursable expenses?",
        expected_output="Employees should submit reimbursable expenses within three months after the date of each expense.",
    ),
    Golden(
        input="What should employees keep for reimbursable expenses?",
        expected_output="They should keep receipts for all reimbursable expenses.",
    ),
    # ============================================================
    # COMPANY CAR
    # ============================================================
    Golden(
        input="What requirements must an employee meet to receive a company car?",
        expected_output="An employee may receive a company car if it is indispensable to their job or is a benefit attached to the job. They should also have a valid driver's license and a clean driving record for at least two years.",
    ),
    Golden(
        input="Can an employee allow an unauthorized person to drive a company car?",
        expected_output="No, unless an emergency mandates it.",
    ),
    Golden(
        input="What should an employee do after an accident involving a company car?",
        expected_output="Contact HR immediately so the company can contact its insurance provider. The employee should not accept responsibility or guarantee payment to another person without authorization.",
    ),
    # ============================================================
    # COMPANY EQUIPMENT
    # ============================================================
    Golden(
        input="Who owns company-issued equipment?",
        expected_output="Unless the contract says otherwise, company-issued equipment belongs to the company and may not be sold or given away.",
    ),
    Golden(
        input="How quickly must stolen or damaged company equipment be reported?",
        expected_output="The company asks employees to report stolen or damaged equipment within 24 hours and, for theft, file a police theft statement/affidavit and submit a copy to the company.",
    ),
    Golden(
        input="How should employees secure company-issued devices?",
        expected_output="Keep devices password-protected and unattended only when secured; install security updates promptly; access company systems only through secure, private networks; and follow instructions for disk encryption, anti-malware protection, and password management.",
    ),
    # ============================================================
    # WORKING HOURS
    # ============================================================
    Golden(
        input="What are the company's normal operating hours?",
        expected_output="The company operates from 9 a.m. to 7 p.m. on weekdays, according to the handbook.",
    ),
    Golden(
        input="When can employees normally arrive at work?",
        expected_output="Employees may normally arrive any time between 9 a.m. and 11 a.m., depending on their team's needs.",
    ),
    # ============================================================
    # PTO
    # ============================================================
    Golden(
        input="How many paid time off days do employees receive per year?",
        expected_output="Employees receive 20 days of PTO per year.",
    ),
    Golden(
        input="How much PTO do employees accrue each month?",
        expected_output="Employees accrue 1.7 days of PTO per month.",
    ),
    Golden(
        input="When does PTO accrual begin?",
        expected_output="PTO accrual begins on the day the employee joins the company.",
    ),
    Golden(
        input="When can a new employee first take PTO?",
        expected_output="A new employee can take PTO after their first week with the company, subject to approval.",
    ),
    Golden(
        input="What is the maximum PTO entitlement?",
        expected_output="The maximum PTO entitlement is 25 days overall.",
    ),
    Golden(
        input="Does an employee need to provide a reason when requesting PTO?",
        expected_output="No. The handbook says employees do not have to specify a reason when requesting PTO.",
    ),
    # ============================================================
    # HOLIDAYS
    # ============================================================
    Golden(
        input="What happens if a company holiday falls on a non-working day?",
        expected_output="The company observes the holiday on the closest business day.",
    ),
    Golden(
        input="Does the company provide a floating holiday?",
        expected_output="Yes. The company offers a floating day that an employee can take as a holiday on any day they choose.",
    ),
    Golden(
        input="How much advance notice is required when an employee must work on a holiday?",
        expected_output="The employee should be informed at least three days in advance.",
    ),
    Golden(
        input="What happens to an exempt employee who works on a holiday?",
        expected_output="The company grants an additional day of PTO, which must be taken within 12 months after the holiday.",
    ),
    # ============================================================
    # SICK LEAVE
    # ============================================================
    Golden(
        input="How much paid sick leave does the company provide?",
        expected_output="The handbook provides one week of paid sick leave, subject to greater leave entitlements required by applicable law.",
    ),
    Golden(
        input="What can sick leave be used for?",
        expected_output="Sick leave can be used to recover from short-term illness, injuries, mental issues, and other indisposition. Employees with flu or another contagious disease are encouraged to use sick leave.",
    ),
    Golden(
        input="When may the company ask for a medical certificate for sick leave?",
        expected_output="The company may ask for a physician's note or other medical certification and/or a sick-leave form when an employee is absent for more than three days of sick leave.",
    ),
    Golden(
        input="What should an employee do when they become sick?",
        expected_output="Inform their manager and submit a sick-leave request through the HRIS. A partial day off or working from home may be possible, but the handbook advises resting and recuperating for a day before returning.",
    ),
    # ============================================================
    # BEREAVEMENT
    # ============================================================
    Golden(
        input="How many days of paid bereavement leave are provided?",
        expected_output="The company provides three days of paid bereavement leave.",
    ),
    Golden(
        input="What can bereavement leave be used for?",
        expected_output="It can be used to arrange or attend a funeral/memorial service, resolve inheritance matters, fulfill other family obligations, and mourn.",
    ),
    Golden(
        input="Can an employee receive additional time off for long-distance travel for a funeral?",
        expected_output="Yes. An employee can take two additional unpaid days for long-distance travel for a funeral or service. Additional time beyond that should be taken as PTO.",
    ),
    # ============================================================
    # JURY DUTY AND VOTING
    # ============================================================
    Golden(
        input="How much time can employees take off to vote?",
        expected_output="On election day, employees can take two hours off to vote. A paid half-day may be available if a short-distance trip is needed; longer travel requires PTO. Hourly employees may take one unpaid day for jury duty and voting, subject to applicable law.",
    ),
    Golden(
        input="What documentation should employees provide for jury duty?",
        expected_output="A copy of the jury-duty summons and a document proving that the employee served.",
    ),
    # ============================================================
    # PARENTAL LEAVE
    # ============================================================
    Golden(
        input="How much paid maternity and paternity leave does the company offer?",
        expected_output="The company offers three months of paid maternity and paternity leave, unless applicable local or national law provides a longer leave.",
    ),
    Golden(
        input="How much notice should new parents give before parental leave?",
        expected_output="New parents should give HR at least three months' notice before parental leave begins.",
    ),
    Golden(
        input="Can parental leave be extended for complications after childbirth?",
        expected_output="Yes. Depending on local or national law, an employee with childbirth complications or other issues may request up to two months of unpaid leave extension.",
    ),
    Golden(
        input="What support is available when employees return from parental leave?",
        expected_output="The handbook lists remote working/flexible hours, onsite/external paid day care, and lactation rooms.",
    ),
    # ============================================================
    # PROGRESSIVE DISCIPLINE
    # ============================================================
    Golden(
        input="What are the six steps in the progressive discipline process?",
        expected_output="The six steps are: (1) verbal warning, (2) informal meeting with supervisor, (3) formal reprimand, (4) formal disciplinary meeting, (5) penalties, and (6) termination.",
    ),
    Golden(
        input="Can managers skip steps in the progressive discipline process?",
        expected_output="Yes. Managers may skip or repeat steps at their discretion, while the company must still act fairly and lawfully and document every stage.",
    ),
    Golden(
        input="What type of offense may trigger the first step of progressive discipline?",
        expected_output="Minor, one-time offenses, such as a breach of the dress code policy, may trigger Step 1 (a verbal warning).",
    ),
    Golden(
        input="Can serious offenses result in termination without a warning?",
        expected_output="Yes. The company may terminate an employee without warning for serious offenses, such as sexual harassment.",
    ),
    # ============================================================
    # RESIGNATION
    # ============================================================
    Golden(
        input="How long can an employee be absent without notice before the company considers them resigned?",
        expected_output="Three consecutive days without notice.",
    ),
    Golden(
        input="How much notice does the company ask employees to give before resigning?",
        expected_output="The company asks for at least two weeks' notice when possible, or at least one month's notice for highly specialized or executive positions.",
    ),
    Golden(
        input="Does the company require advance notice before resignation?",
        expected_output="No. Employees are not obliged to give advance notice, although the company asks for two weeks' notice when possible (or one month for highly specialized/executive positions).",
    ),
    Golden(
        input="Does the company accept verbal resignations?",
        expected_output="Yes. Verbal resignations are accepted, although the company prefers a written and signed notice for HR records.",
    ),
    # ============================================================
    # TUITION / RELOCATION
    # ============================================================
    Golden(
        input="How long must an employee remain with the company after receiving tuition or relocation support?",
        expected_output="The employee is contractually required to remain with the company for at least two years.",
    ),
    Golden(
        input="What happens if an employee resigns before completing the required period after company-funded relocation or study?",
        expected_output="They may have to reimburse the company for part or all of the tuition or relocation expenses.",
    ),
    # ============================================================
    # TERMINATION
    # ============================================================
    Golden(
        input="What are the reasons an employee may be terminated for cause?",
        expected_output="For-cause termination may result from breaching the contract, engaging in illegal activities such as embezzlement, disrupting the workplace such as by harassing colleagues, performing below acceptable standards, or causing company damage or financial loss.",
    ),
    Golden(
        input="What does termination without cause mean?",
        expected_output="It refers to redundancies or layoffs that may be necessary when the company ceases some operations or reassigns job duties within teams. Applicable laws regarding notice and payouts will be followed.",
    ),
    Golden(
        input="Does the company provide severance pay to terminated employees?",
        expected_output="Yes. The company states that it will offer severance pay to eligible employees.",
    ),
    Golden(
        input="Can accrued leave be compensated when an employee is terminated?",
        expected_output="Yes, accrued vacation and sick leave may be compensated upon termination depending on local law. Where local law has no relevant provision, accrued leave is paid only to employees who were not terminated for cause, subject also to union agreements.",
    ),
    # ============================================================
    # REFERENCES
    # ============================================================
    Golden(
        input="Can employees receive references after leaving the company?",
        expected_output="Possibly. Employees leaving in good standing may receive references; laid-off employees may also receive references. Employees who resign may ask, but their manager may accept or refuse the request.",
    ),
    # ============================================================
    # HANDBOOK / POLICY
    # ============================================================
    Golden(
        input="Is the employee handbook a contract?",
        expected_output="No. The handbook is not a contract or a guarantee of employment; it is a collection of expectations, commitments, and responsibilities.",
    ),
    Golden(
        input="How often is the employee handbook reviewed?",
        expected_output="The handbook is reviewed annually to keep it up to date with legislation and employment trends.",
    ),
    Golden(
        input="What should employees do if they find an inconsistency or mistake in the handbook?",
        expected_output="Contact HR to report the inconsistency or mistake. Employees may also share ideas for improving the workplace.",
    ),
]
