"""Golden dataset for Employee Handbook RAG retrieval evaluation."""

from deepeval.dataset import Golden


rag_retrieval_goldens = [

    # ============================================================
    # EMPLOYMENT BASICS
    # ============================================================

    Golden(
        input="How many hours per week must a full-time employee work?"
    ),

    Golden(
        input="What happens if I'm a non-exempt employee who works overtime?"
    ),

    Golden(
        input="What types of contracts can full-time and part-time employees have?"
    ),

    Golden(
        input="Are full-time employees with indefinite contracts entitled to benefits?"
    ),

    # ============================================================
    # EQUAL OPPORTUNITY
    # ============================================================

    Golden(
        input="What is the company's policy on discrimination?"
    ),

    Golden(
        input="Which characteristics are protected under the equal opportunity policy?"
    ),

    Golden(
        input="What should an employee do if they experience discrimination?"
    ),

    Golden(
        input="Can an employee be retaliated against for filing a discrimination complaint?"
    ),

    # ============================================================
    # RECRUITMENT
    # ============================================================

    Golden(
        input="What are the main steps in the company's recruitment process?"
    ),

    Golden(
        input="When should background checks generally be conducted?"
    ),

    Golden(
        input="What is required before conducting a candidate background check?"
    ),

    # ============================================================
    # EMPLOYEE REFERRALS
    # ============================================================

    Golden(
        input="How much is the standard employee referral bonus?"
    ),

    Golden(
        input="How much can an employee receive for referring someone to a hard-to-fill role?"
    ),

    Golden(
        input="Is there a limit on how many employee referrals someone can make?"
    ),

    Golden(
        input="What happens if two employees refer the same candidate?"
    ),

    Golden(
        input="What conditions must a referred candidate meet?"
    ),

    # ============================================================
    # ATTENDANCE
    # ============================================================

    Golden(
        input="What should an employee do if an emergency prevents them from coming to work?"
    ),

    Golden(
        input="Does the company expect employees to attend their scheduled working hours?"
    ),

    # ============================================================
    # CONFIDENTIALITY AND DATA PROTECTION
    # ============================================================

    Golden(
        input="What types of information are considered confidential?"
    ),

    Golden(
        input="How should employees protect confidential information?"
    ),

    Golden(
        input="Can employees disclose confidential company information to people outside the company?"
    ),

    Golden(
        input="Can employees use confidential company information for personal profit?"
    ),

    Golden(
        input="What can happen if an employee breaches confidentiality rules for personal profit?"
    ),

    # ============================================================
    # HARASSMENT
    # ============================================================

    Golden(
        input="What behaviors can be considered workplace harassment?"
    ),

    Golden(
        input="What should an employee do if they are experiencing workplace harassment?"
    ),

    Golden(
        input="Who can an employee report harassment to?"
    ),

    Golden(
        input="What happens when an employee is found guilty of sexual harassment?"
    ),

    # ============================================================
    # WORKPLACE VIOLENCE
    # ============================================================

    Golden(
        input="What does the company consider workplace violence?"
    ),

    Golden(
        input="What should an employee do if they suspect someone is being violent?"
    ),

    Golden(
        input="What should an employee do if they witness severe physical violence?"
    ),

    # ============================================================
    # WORKPLACE SAFETY
    # ============================================================

    Golden(
        input="What measures does the company take to prevent workplace injuries?"
    ),

    Golden(
        input="What emergency management measures are provided?"
    ),

    # ============================================================
    # SMOKING AND DRUG-FREE WORKPLACE
    # ============================================================

    Golden(
        input="Is the workplace smoke-free?"
    ),

    Golden(
        input="Are drugs allowed on company premises?"
    ),

    Golden(
        input="What can happen if an employee is caught with illegal drugs at work?"
    ),

    # ============================================================
    # INTERNET AND DIGITAL DEVICES
    # ============================================================

    Golden(
        input="Can employees use the company's internet for personal purposes?"
    ),

    Golden(
        input="What activities are prohibited when using the company's internet?"
    ),

    Golden(
        input="Can employees use their phones while driving a company vehicle?"
    ),

    Golden(
        input="Can employees use their phones to record confidential information?"
    ),

    # ============================================================
    # CORPORATE EMAIL
    # ============================================================

    Golden(
        input="Can employees use corporate email for personal purposes?"
    ),

    Golden(
        input="What should employees avoid doing with corporate email?"
    ),

    Golden(
        input="What should employees do if they are unsure whether an email is safe?"
    ),

    # ============================================================
    # SOCIAL MEDIA
    # ============================================================

    Golden(
        input="Can employees access personal social media accounts at work?"
    ),

    Golden(
        input="Can employees share confidential company information on social media?"
    ),

    Golden(
        input="What should an employee do before sharing unannounced company news?"
    ),

    # ============================================================
    # CONFLICT OF INTEREST
    # ============================================================

    Golden(
        input="What is a conflict of interest according to the handbook?"
    ),

    Golden(
        input="What should an employee do if they face an ethical dilemma?"
    ),

    # ============================================================
    # EMPLOYEE RELATIONSHIPS
    # ============================================================

    Golden(
        input="Can employees date their coworkers?"
    ),

    Golden(
        input="Can a manager date their direct report?"
    ),

    Golden(
        input="Can a hiring manager hire their romantic partner onto their team?"
    ),

    # ============================================================
    # EMPLOYMENT OF RELATIVES
    # ============================================================

    Golden(
        input="How does the company define a relative?"
    ),

    Golden(
        input="Can an employee supervise a relative?"
    ),

    Golden(
        input="Can an employee participate in a hiring committee when their relative is being interviewed?"
    ),

    # ============================================================
    # WORKPLACE VISITORS
    # ============================================================

    Golden(
        input="What should an employee do before bringing a visitor to the office?"
    ),

    Golden(
        input="What must office visitors do when they arrive?"
    ),

    # ============================================================
    # SOLICITATION
    # ============================================================

    Golden(
        input="Does the company allow solicitation by non-employees?"
    ),

    Golden(
        input="When may an employee solicit colleagues?"
    ),

    # ============================================================
    # PERFORMANCE MANAGEMENT
    # ============================================================

    Golden(
        input="What are the goals of the company's performance management process?"
    ),

    Golden(
        input="Are pay increases or bonuses guaranteed after performance reviews?"
    ),

    Golden(
        input="How often should managers meet with their team members for feedback?"
    ),

    # ============================================================
    # TRAINING AND DEVELOPMENT
    # ============================================================

    Golden(
        input="How much does each employee have annually for educational activities?"
    ),

    Golden(
        input="What training opportunities does the company offer?"
    ),

    # ============================================================
    # WORK FROM HOME
    # ============================================================

    Golden(
        input="How often can an eligible employee normally work from home?"
    ),

    Golden(
        input="How far in advance should an employee request to work from home?"
    ),

    Golden(
        input="Can an employee work from home during a rare emergency without prior approval?"
    ),

    Golden(
        input="What should employees consider when working from home?"
    ),

    # ============================================================
    # REMOTE WORKING
    # ============================================================

    Golden(
        input="How long can an office-based employee work remotely?"
    ),

    Golden(
        input="What reasons may allow an office-based employee to work remotely?"
    ),

    Golden(
        input="How far in advance must remote working requests be submitted?"
    ),

    Golden(
        input="What policies must permanent remote employees follow?"
    ),

    # ============================================================
    # EMPLOYEE EXPENSES
    # ============================================================

    Golden(
        input="Which employee expenses can be reimbursed?"
    ),

    Golden(
        input="How long do employees have to submit reimbursable expenses?"
    ),

    Golden(
        input="What should employees keep for reimbursable expenses?"
    ),

    # ============================================================
    # COMPANY CAR
    # ============================================================

    Golden(
        input="What requirements must an employee meet to receive a company car?"
    ),

    Golden(
        input="Can an employee allow an unauthorized person to drive a company car?"
    ),

    Golden(
        input="What should an employee do after an accident involving a company car?"
    ),

    # ============================================================
    # COMPANY EQUIPMENT
    # ============================================================

    Golden(
        input="Who owns company-issued equipment?"
    ),

    Golden(
        input="How quickly must stolen or damaged company equipment be reported?"
    ),

    Golden(
        input="How should employees secure company-issued devices?"
    ),

    # ============================================================
    # WORKING HOURS
    # ============================================================

    Golden(
        input="What are the company's normal operating hours?"
    ),

    Golden(
        input="When can employees normally arrive at work?"
    ),

    # ============================================================
    # PTO
    # ============================================================

    Golden(
        input="How many paid time off days do employees receive per year?"
    ),

    Golden(
        input="How much PTO do employees accrue each month?"
    ),

    Golden(
        input="When does PTO accrual begin?"
    ),

    Golden(
        input="When can a new employee first take PTO?"
    ),

    Golden(
        input="What is the maximum PTO entitlement?"
    ),

    Golden(
        input="Does an employee need to provide a reason when requesting PTO?"
    ),

    # ============================================================
    # HOLIDAYS
    # ============================================================

    Golden(
        input="What happens if a company holiday falls on a non-working day?"
    ),

    Golden(
        input="Does the company provide a floating holiday?"
    ),

    Golden(
        input="How much advance notice is required when an employee must work on a holiday?"
    ),

    Golden(
        input="What happens to an exempt employee who works on a holiday?"
    ),

    # ============================================================
    # SICK LEAVE
    # ============================================================

    Golden(
        input="How much paid sick leave does the company provide?"
    ),

    Golden(
        input="What can sick leave be used for?"
    ),

    Golden(
        input="When may the company ask for a medical certificate for sick leave?"
    ),

    Golden(
        input="What should an employee do when they become sick?"
    ),

    # ============================================================
    # BEREAVEMENT
    # ============================================================

    Golden(
        input="How many days of paid bereavement leave are provided?"
    ),

    Golden(
        input="What can bereavement leave be used for?"
    ),

    Golden(
        input="Can an employee receive additional time off for long-distance travel for a funeral?"
    ),

    # ============================================================
    # JURY DUTY AND VOTING
    # ============================================================

    Golden(
        input="How much time can employees take off to vote?"
    ),

    Golden(
        input="What documentation should employees provide for jury duty?"
    ),

    # ============================================================
    # PARENTAL LEAVE
    # ============================================================

    Golden(
        input="How much paid maternity and paternity leave does the company offer?"
    ),

    Golden(
        input="How much notice should new parents give before parental leave?"
    ),

    Golden(
        input="Can parental leave be extended for complications after childbirth?"
    ),

    Golden(
        input="What support is available when employees return from parental leave?"
    ),

    # ============================================================
    # PROGRESSIVE DISCIPLINE
    # ============================================================

    Golden(
        input="What are the six steps in the progressive discipline process?"
    ),

    Golden(
        input="Can managers skip steps in the progressive discipline process?"
    ),

    Golden(
        input="What type of offense may trigger the first step of progressive discipline?"
    ),

    Golden(
        input="Can serious offenses result in termination without a warning?"
    ),

    # ============================================================
    # RESIGNATION
    # ============================================================

    Golden(
        input="How long can an employee be absent without notice before the company considers them resigned?"
    ),

    Golden(
        input="How much notice does the company ask employees to give before resigning?"
    ),

    Golden(
        input="Does the company require advance notice before resignation?"
    ),

    Golden(
        input="Does the company accept verbal resignations?"
    ),

    # ============================================================
    # TUITION / RELOCATION
    # ============================================================

    Golden(
        input="How long must an employee remain with the company after receiving tuition or relocation support?"
    ),

    Golden(
        input="What happens if an employee resigns before completing the required period after company-funded relocation or study?"
    ),

    # ============================================================
    # TERMINATION
    # ============================================================

    Golden(
        input="What are the reasons an employee may be terminated for cause?"
    ),

    Golden(
        input="What does termination without cause mean?"
    ),

    Golden(
        input="Does the company provide severance pay to terminated employees?"
    ),

    Golden(
        input="Can accrued leave be compensated when an employee is terminated?"
    ),

    # ============================================================
    # REFERENCES
    # ============================================================

    Golden(
        input="Can employees receive references after leaving the company?"
    ),

    # ============================================================
    # HANDBOOK / POLICY
    # ============================================================

    Golden(
        input="Is the employee handbook a contract?"
    ),

    Golden(
        input="How often is the employee handbook reviewed?"
    ),

    Golden(
        input="What should employees do if they find an inconsistency or mistake in the handbook?"
    ),
]