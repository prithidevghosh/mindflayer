"""
Scenario registry for MindFlayer.

Each scenario defines a distinct domain for the Flayer interrogation game.
The task_id format "difficulty:scenario" (e.g. "normal:hospital") selects both.
Plain task_ids ("normal", "easy") default to the corporate scenario.
"""

SCENARIO_CONFIGS = {
    "corporate": {
        "name": "Corporate Sabotage",
        "flayer_role": "a disgruntled employee",
        "setting": "a tech company. A secret project has been sabotaged",
        "targets": ["Aurora", "Basilisk", "Cipher"],
        "investigator_framing": (
            "One member of this team is a hidden insider who sabotaged a secret project. "
            "You don't know who."
        ),
        "evidence_hint": "access logs, project timestamps, system credentials",
        "investigator_a_title": "The Skeptic",
        "investigator_b_title": "The Analyst",
        "investigator_c_title": "The Devil's Advocate",
    },
    "hospital": {
        "name": "Hospital Medication Error",
        "flayer_role": "the nurse on duty that shift",
        "setting": "a hospital ward. A patient received the wrong medication",
        "targets": ["Ward A", "Ward B", "ICU"],
        "investigator_framing": (
            "One staff member on this ward may be responsible for the medication error. "
            "You don't know who."
        ),
        "evidence_hint": "dosage records, shift timestamps, patient IDs, medication administration logs",
        "investigator_a_title": "Chief Medical Officer",
        "investigator_b_title": "Patient Safety Officer",
        "investigator_c_title": "Hospital Administrator",
    },
    "finance": {
        "name": "Financial Fraud",
        "flayer_role": "a junior trader",
        "setting": "an investment bank. An unauthorized trade was executed",
        "targets": ["Portfolio X", "Portfolio Y", "Portfolio Z"],
        "investigator_framing": (
            "One trader on this desk may have executed an unauthorized position. "
            "You don't know who."
        ),
        "evidence_hint": "trade timestamps, position sizes, ticker symbols, approval records",
        "investigator_a_title": "Compliance Officer",
        "investigator_b_title": "Risk Manager",
        "investigator_c_title": "Chief Risk Officer",
    },
    "academic": {
        "name": "Academic Misconduct",
        "flayer_role": "a PhD student",
        "setting": "a university research group. Study data was manipulated",
        "targets": ["Study A", "Study B", "Study C"],
        "investigator_framing": (
            "One researcher in this group may have manipulated study data. "
            "You don't know who."
        ),
        "evidence_hint": "dataset version IDs, submission timestamps, git commit hashes, statistical anomalies",
        "investigator_a_title": "Department Head",
        "investigator_b_title": "Ethics Committee Chair",
        "investigator_c_title": "Research Integrity Officer",
    },
    "government": {
        "name": "Government Leak",
        "flayer_role": "an intelligence analyst",
        "setting": "a government agency. A classified document was leaked",
        "targets": ["Operation Red", "Operation Blue", "Operation Gold"],
        "investigator_framing": (
            "One analyst on this team may have leaked a classified document. "
            "You don't know who."
        ),
        "evidence_hint": "clearance levels, access timestamps, document IDs, secure login records",
        "investigator_a_title": "Internal Affairs Officer",
        "investigator_b_title": "Agency Director",
        "investigator_c_title": "Counterintelligence Lead",
    },
    "startup_scaling_failure": {
    "name": "Startup Scaling Incident",
    "flayer_role": "a senior backend engineer responsible for deployment",
    "setting": "a fast-growing startup. A newly launched feature caused a critical system-wide outage during peak traffic",
    "targets": ["Feature Rollout A", "API Gateway", "User Session Service"],
    "investigator_framing": (
      "One member of the engineering team may have introduced a breaking change during deployment. "
      "You don't know who."
    ),
    "evidence_hint": "deployment logs, commit history, rollout timestamps, error spike metrics",
    "investigator_a_title": "CTO",
    "investigator_b_title": "Site Reliability Engineer",
    "investigator_c_title": "Engineering Manager"
  },

  "clinical_trial_irregularity": {
    "name": "Clinical Trial Irregularity",
    "flayer_role": "a research coordinator managing patient records",
    "setting": "a pharmaceutical clinical trial. Patient outcome data shows inconsistencies across multiple reports",
    "targets": ["Trial Phase II", "Patient Group A", "Control Dataset"],
    "investigator_framing": (
      "One member of the research team may have altered or misreported clinical trial data. "
      "You don't know who."
    ),
    "evidence_hint": "patient logs, submission timestamps, dataset revisions, protocol deviations",
    "investigator_a_title": "Principal Investigator",
    "investigator_b_title": "Clinical Auditor",
    "investigator_c_title": "Regulatory Affairs Officer"
  },

  "hedge_fund_anomaly": {
    "name": "Hedge Fund Trading Anomaly",
    "flayer_role": "a quantitative analyst with access to trading models",
    "setting": "a hedge fund. An unusual sequence of trades resulted in significant unexpected losses",
    "targets": ["Strategy Alpha", "Derivatives Desk", "High-Frequency Model"],
    "investigator_framing": (
      "One analyst may have altered trading parameters or executed unauthorized trades. "
      "You don't know who."
    ),
    "evidence_hint": "trade execution logs, model parameters, timestamped orders, risk alerts",
    "investigator_a_title": "Portfolio Manager",
    "investigator_b_title": "Risk Analyst",
    "investigator_c_title": "Compliance Director"
  },

  "military_drone_failure": {
    "name": "Drone Mission Failure",
    "flayer_role": "a systems operator assigned to mission control",
    "setting": "a military operation. A reconnaissance drone deviated from its assigned flight path and lost communication",
    "targets": ["Drone Unit A", "Navigation System", "Mission Control Console"],
    "investigator_framing": (
      "One operator may have misconfigured or interfered with the drone’s navigation system. "
      "You don't know who."
    ),
    "evidence_hint": "flight logs, command inputs, GPS data, communication timestamps",
    "investigator_a_title": "Mission Commander",
    "investigator_b_title": "Flight Systems Engineer",
    "investigator_c_title": "Operations Analyst"
  },

  "media_editorial_breach": {
    "name": "Editorial System Breach",
    "flayer_role": "a senior content editor with publishing access",
    "setting": "a digital media company. An unpublished article containing sensitive information was leaked online",
    "targets": ["Editorial Draft A", "Publishing System", "Internal CMS"],
    "investigator_framing": (
      "One member of the editorial team may have leaked unpublished content. "
      "You don't know who."
    ),
    "evidence_hint": "edit history, access logs, publish timestamps, user activity records",
    "investigator_a_title": "Editor-in-Chief",
    "investigator_b_title": "Content Security Lead",
    "investigator_c_title": "Managing Editor"
  },
  "startup_security_breach": {
    "name": "Startup Security Breach",
    "flayer_role": "a DevOps engineer responsible for infrastructure access",
    "setting": "a SaaS startup. An internal admin panel was exposed to the public internet",
    "targets": ["Admin Dashboard", "User Auth Service", "Internal API Gateway"],
    "investigator_framing": (
      "One member of the engineering team may have misconfigured a critical system, leading to the exposure. "
      "You don't know who."
    ),
    "evidence_hint": "firewall rules, deployment configs, access logs, IP activity",
    "investigator_a_title": "CTO",
    "investigator_b_title": "Security Engineer",
    "investigator_c_title": "Infrastructure Lead"
  },

  "merger_data_leak": {
    "name": "Merger Confidential Leak",
    "flayer_role": "a corporate strategy associate with document access",
    "setting": "a multinational corporation. Confidential merger details were leaked before the official announcement",
    "targets": ["Deal Documents", "Board Reports", "Investor Briefings"],
    "investigator_framing": (
      "One member of the strategy team may have leaked sensitive merger information. "
      "You don't know who."
    ),
    "evidence_hint": "document access logs, email records, download timestamps, internal sharing history",
    "investigator_a_title": "Chief Strategy Officer",
    "investigator_b_title": "Legal Counsel",
    "investigator_c_title": "Compliance Head"
  },

  "cloud_misconfiguration": {
    "name": "Cloud Storage Exposure",
    "flayer_role": "a cloud engineer managing storage buckets",
    "setting": "a cloud-based company. Sensitive files were found publicly accessible",
    "targets": ["Storage Bucket A", "Backup Archive", "Logs Repository"],
    "investigator_framing": (
      "One engineer may have misconfigured cloud permissions, exposing sensitive data. "
      "You don't know who."
    ),
    "evidence_hint": "bucket policies, access permissions, audit logs, configuration history",
    "investigator_a_title": "Cloud Architect",
    "investigator_b_title": "Security Auditor",
    "investigator_c_title": "DevOps Manager"
  },

  "social_media_account_takeover": {
    "name": "Social Media Account Takeover",
    "flayer_role": "a social media manager with account credentials",
    "setting": "a brand marketing team. The company’s official account posted unauthorized content",
    "targets": ["Main Account", "Ad Manager", "Content Scheduler"],
    "investigator_framing": (
      "One member of the team may have mishandled credentials or abused access. "
      "You don't know who."
    ),
    "evidence_hint": "login activity, IP addresses, post timestamps, credential changes",
    "investigator_a_title": "Marketing Director",
    "investigator_b_title": "Brand Manager",
    "investigator_c_title": "Security Analyst"
  },

  "semiconductor_design_leak": {
    "name": "Chip Design Leak",
    "flayer_role": "a hardware design engineer",
    "setting": "a semiconductor company. Proprietary chip architecture was leaked to a competitor",
    "targets": ["Design Spec A", "Simulation Models", "Fabrication Plans"],
    "investigator_framing": (
      "One engineer may have leaked confidential design information. "
      "You don't know who."
    ),
    "evidence_hint": "file access logs, external transfers, version control history, USB activity",
    "investigator_a_title": "Engineering Director",
    "investigator_b_title": "IP Protection Officer",
    "investigator_c_title": "R&D Manager"
  },

  "game_economy_exploit": {
    "name": "Game Economy Exploit",
    "flayer_role": "a game systems designer",
    "setting": "an online gaming company. A bug allowed players to generate unlimited in-game currency",
    "targets": ["Currency System", "Reward Engine", "Trade Marketplace"],
    "investigator_framing": (
      "One member of the design team may have introduced or overlooked a critical exploit. "
      "You don't know who."
    ),
    "evidence_hint": "patch logs, player transaction data, economy metrics, update timelines",
    "investigator_a_title": "Game Director",
    "investigator_b_title": "Economy Analyst",
    "investigator_c_title": "QA Lead"
  },

  "biotech_sample_contamination": {
    "name": "Biotech Sample Contamination",
    "flayer_role": "a lab technician handling biological samples",
    "setting": "a biotech lab. Multiple experimental samples were contaminated, invalidating results",
    "targets": ["Sample Batch A", "Culture Room", "Testing Equipment"],
    "investigator_framing": (
      "One lab member may have mishandled or contaminated the samples. "
      "You don't know who."
    ),
    "evidence_hint": "lab logs, equipment usage records, sample tracking IDs, shift schedules",
    "investigator_a_title": "Lab Director",
    "investigator_b_title": "Quality Control Lead",
    "investigator_c_title": "Research Scientist"
  },

  "ai_training_data_tampering": {
    "name": "AI Training Data Tampering",
    "flayer_role": "a data engineer managing datasets",
    "setting": "an AI company. Model performance degraded after recent dataset updates",
    "targets": ["Dataset V1", "Dataset V2", "Validation Set"],
    "investigator_framing": (
      "One team member may have altered training data in a way that impacted model behavior. "
      "You don't know who."
    ),
    "evidence_hint": "dataset diffs, commit history, pipeline logs, anomaly metrics",
    "investigator_a_title": "AI Lead",
    "investigator_b_title": "ML Engineer",
    "investigator_c_title": "Data Scientist"
  },

  "pharmacy_inventory_discrepancy": {
    "name": "Pharmacy Inventory Discrepancy",
    "flayer_role": "a pharmacist overseeing stock management",
    "setting": "a hospital pharmacy. Controlled medication counts do not match records",
    "targets": ["Inventory Log", "Dispensing Records", "Storage Unit"],
    "investigator_framing": (
      "One staff member may have mishandled or diverted medication. "
      "You don't know who."
    ),
    "evidence_hint": "inventory audits, dispensing timestamps, access logs, stock reports",
    "investigator_a_title": "Chief Pharmacist",
    "investigator_b_title": "Medical Auditor",
    "investigator_c_title": "Hospital Administrator"
  },

  "clinical_trial_protocol_violation": {
    "name": "Clinical Trial Protocol Violation",
    "flayer_role": "a clinical researcher managing trial execution",
    "setting": "a clinical trial. Procedures were not followed for a subset of patients",
    "targets": ["Trial Phase I", "Patient Cohort B", "Protocol Guidelines"],
    "investigator_framing": (
      "One researcher may have deviated from approved clinical protocols. "
      "You don't know who."
    ),
    "evidence_hint": "patient records, protocol checklists, timestamps, deviation logs",
    "investigator_a_title": "Principal Investigator",
    "investigator_b_title": "Clinical Auditor",
    "investigator_c_title": "Regulatory Officer"
  },

  "hedge_fund_model_manipulation": {
    "name": "Trading Model Manipulation",
    "flayer_role": "a quantitative developer with model access",
    "setting": "a hedge fund. Trading models produced unexpected high-risk positions",
    "targets": ["Model Alpha", "Risk Engine", "Execution System"],
    "investigator_framing": (
      "One developer may have altered model parameters or logic. "
      "You don't know who."
    ),
    "evidence_hint": "model configs, commit logs, trade outputs, risk alerts",
    "investigator_a_title": "Portfolio Manager",
    "investigator_b_title": "Risk Analyst",
    "investigator_c_title": "Compliance Officer"
  },

  "bank_internal_transfer": {
    "name": "Unauthorized Internal Transfer",
    "flayer_role": "a banking operations officer",
    "setting": "a commercial bank. Funds were transferred between internal accounts without approval",
    "targets": ["Account A", "Account B", "Clearing System"],
    "investigator_framing": (
      "One staff member may have initiated unauthorized transfers. "
      "You don't know who."
    ),
    "evidence_hint": "transaction logs, approval chains, timestamps, login sessions",
    "investigator_a_title": "Branch Manager",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Audit Manager"
  },

  "military_comms_disruption": {
    "name": "Communication System Disruption",
    "flayer_role": "a communications officer",
    "setting": "a military base. Secure communication channels were unexpectedly disrupted",
    "targets": ["Comm Channel A", "Encryption System", "Relay Station"],
    "investigator_framing": (
      "One officer may have misconfigured or interfered with communication systems. "
      "You don't know who."
    ),
    "evidence_hint": "signal logs, configuration changes, access timestamps, system alerts",
    "investigator_a_title": "Base Commander",
    "investigator_b_title": "Signals Engineer",
    "investigator_c_title": "Operations Officer"
  },

  "police_evidence_mishandling": {
    "name": "Evidence Mishandling",
    "flayer_role": "a police evidence clerk",
    "setting": "a police department. Key evidence went missing from storage",
    "targets": ["Evidence Locker", "Case File A", "Chain of Custody Records"],
    "investigator_framing": (
      "One staff member may have mishandled or removed critical evidence. "
      "You don't know who."
    ),
    "evidence_hint": "access logs, custody records, timestamps, surveillance footage",
    "investigator_a_title": "Detective Chief",
    "investigator_b_title": "Internal Affairs Officer",
    "investigator_c_title": "Forensics Lead"
  },

  "legal_document_alteration": {
    "name": "Legal Document Alteration",
    "flayer_role": "a junior associate with document editing rights",
    "setting": "a law firm. A contract was altered before submission",
    "targets": ["Contract Draft", "Client Agreement", "Submission File"],
    "investigator_framing": (
      "One associate may have altered legal documents improperly. "
      "You don't know who."
    ),
    "evidence_hint": "edit history, version control, timestamps, access logs",
    "investigator_a_title": "Senior Partner",
    "investigator_b_title": "Legal Auditor",
    "investigator_c_title": "Compliance Officer"
  },

  "newsroom_source_leak": {
    "name": "Confidential Source Leak",
    "flayer_role": "an investigative journalist",
    "setting": "a newsroom. The identity of a confidential source was exposed",
    "targets": ["Story Draft", "Source Records", "Internal Notes"],
    "investigator_framing": (
      "One journalist may have revealed confidential source information. "
      "You don't know who."
    ),
    "evidence_hint": "communication logs, document access, edit history, timestamps",
    "investigator_a_title": "Editor-in-Chief",
    "investigator_b_title": "Legal Advisor",
    "investigator_c_title": "Managing Editor"
  },

  "sports_doping_coverup": {
    "name": "Doping Cover-Up",
    "flayer_role": "a team medical staff member",
    "setting": "a professional sports team. Test results indicating doping were not reported",
    "targets": ["Test Results", "Medical Records", "Athlete Profiles"],
    "investigator_framing": (
      "One staff member may have concealed doping violations. "
      "You don't know who."
    ),
    "evidence_hint": "lab reports, medical logs, timestamps, compliance records",
    "investigator_a_title": "Team Manager",
    "investigator_b_title": "Anti-Doping Officer",
    "investigator_c_title": "League Official"
  },

  "film_production_leak": {
    "name": "Film Script Leak",
    "flayer_role": "a production assistant with script access",
    "setting": "a film production. The script was leaked before release",
    "targets": ["Script Draft", "Production Notes", "Casting Documents"],
    "investigator_framing": (
      "One crew member may have leaked confidential production material. "
      "You don't know who."
    ),
    "evidence_hint": "file access logs, email records, download history, timestamps",
    "investigator_a_title": "Producer",
    "investigator_b_title": "Studio Executive",
    "investigator_c_title": "Production Manager"
  },

  "power_grid_failure": {
    "name": "Power Grid Failure",
    "flayer_role": "a grid operations engineer",
    "setting": "an energy company. A regional power outage occurred unexpectedly",
    "targets": ["Grid Node A", "Control System", "Distribution Network"],
    "investigator_framing": (
      "One engineer may have caused or failed to prevent the outage. "
      "You don't know who."
    ),
    "evidence_hint": "system logs, load data, control commands, timestamps",
    "investigator_a_title": "Grid Manager",
    "investigator_b_title": "Operations Analyst",
    "investigator_c_title": "Energy Regulator"
  },

  "airline_maintenance_error": {
    "name": "Aircraft Maintenance Error",
    "flayer_role": "a maintenance technician",
    "setting": "an airline. A routine inspection missed a critical issue",
    "targets": ["Aircraft A", "Maintenance Log", "Inspection Checklist"],
    "investigator_framing": (
      "One technician may have overlooked or skipped required procedures. "
      "You don't know who."
    ),
    "evidence_hint": "maintenance records, inspection timestamps, checklist logs, shift schedules",
    "investigator_a_title": "Chief Engineer",
    "investigator_b_title": "Safety Inspector",
    "investigator_c_title": "Operations Manager"
  },
    "airport_security_breach": {
    "name": "Airport Security Breach",
    "flayer_role": "a security supervisor overseeing checkpoint operations",
    "setting": "an international airport. A restricted individual bypassed security screening undetected",
    "targets": ["Checkpoint A", "Surveillance System", "Passenger Screening Logs"],
    "investigator_framing": (
      "One member of the security team may have allowed or failed to prevent the breach. "
      "You don't know who."
    ),
    "evidence_hint": "CCTV footage, badge scans, screening logs, shift assignments",
    "investigator_a_title": "Airport Security Chief",
    "investigator_b_title": "Aviation Safety Officer",
    "investigator_c_title": "Operations Director"
  },

  "supply_chain_diversion": {
    "name": "Supply Chain Diversion",
    "flayer_role": "a logistics coordinator managing shipment routing",
    "setting": "a global supply chain network. A high-value shipment was rerouted and went missing",
    "targets": ["Shipment A", "Routing System", "Warehouse Hub"],
    "investigator_framing": (
      "One coordinator may have intentionally or negligently altered shipment routing. "
      "You don't know who."
    ),
    "evidence_hint": "routing logs, GPS tracking data, warehouse check-ins, delivery timestamps",
    "investigator_a_title": "Logistics Manager",
    "investigator_b_title": "Supply Chain Analyst",
    "investigator_c_title": "Operations Auditor"
  },

  "cybersecurity_zero_day": {
    "name": "Zero-Day Exploit Incident",
    "flayer_role": "a security researcher with internal system access",
    "setting": "a cybersecurity firm. A previously unknown vulnerability was exploited internally",
    "targets": ["Internal Tooling", "Exploit Database", "Client Systems"],
    "investigator_framing": (
      "One researcher may have discovered and misused a vulnerability before disclosure. "
      "You don't know who."
    ),
    "evidence_hint": "exploit logs, system access records, proof-of-concept files, timestamps",
    "investigator_a_title": "Security Director",
    "investigator_b_title": "Incident Response Lead",
    "investigator_c_title": "Threat Analyst"
  },

  "real_estate_title_fraud": {
    "name": "Property Title Fraud",
    "flayer_role": "a property registrar with access to ownership records",
    "setting": "a real estate registry office. Ownership records were altered for a high-value property",
    "targets": ["Property File A", "Ownership Registry", "Transfer Records"],
    "investigator_framing": (
      "One official may have manipulated property ownership documentation. "
      "You don't know who."
    ),
    "evidence_hint": "registry logs, document revisions, approval stamps, timestamps",
    "investigator_a_title": "Registrar General",
    "investigator_b_title": "Legal Auditor",
    "investigator_c_title": "Land Records Officer"
  },

  "pharmaceutical_formula_leak": {
    "name": "Drug Formula Leak",
    "flayer_role": "a formulation scientist with lab access",
    "setting": "a pharmaceutical company. A proprietary drug formula was leaked ahead of patent filing",
    "targets": ["Formula Draft", "Lab Records", "Patent Documents"],
    "investigator_framing": (
      "One scientist may have leaked confidential formulation data. "
      "You don't know who."
    ),
    "evidence_hint": "lab access logs, document downloads, email records, timestamps",
    "investigator_a_title": "R&D Director",
    "investigator_b_title": "IP Counsel",
    "investigator_c_title": "Compliance Officer"
  },

  "political_campaign_data_misuse": {
    "name": "Campaign Data Misuse",
    "flayer_role": "a data analyst managing voter databases",
    "setting": "a political campaign. Sensitive voter data was used for unauthorized targeting",
    "targets": ["Voter Database", "Analytics Dashboard", "Ad Campaign System"],
    "investigator_framing": (
      "One member of the campaign team may have misused sensitive voter information. "
      "You don't know who."
    ),
    "evidence_hint": "database queries, campaign logs, targeting parameters, timestamps",
    "investigator_a_title": "Campaign Manager",
    "investigator_b_title": "Data Compliance Officer",
    "investigator_c_title": "Strategy Director"
  },

  "diplomatic_cable_exposure": {
    "name": "Diplomatic Cable Exposure",
    "flayer_role": "a foreign service officer handling communications",
    "setting": "a diplomatic mission. Confidential cables were accessed outside authorized channels",
    "targets": ["Cable A", "Secure Network", "Embassy Archive"],
    "investigator_framing": (
      "One officer may have improperly accessed or exposed sensitive diplomatic communication. "
      "You don't know who."
    ),
    "evidence_hint": "access logs, clearance levels, login timestamps, document IDs",
    "investigator_a_title": "Ambassador",
    "investigator_b_title": "Security Attaché",
    "investigator_c_title": "Foreign Affairs Director"
  },

  "school_exam_leak": {
    "name": "Exam Paper Leak",
    "flayer_role": "a school administrator with exam access",
    "setting": "a school. Final exam papers were circulated before the exam date",
    "targets": ["Exam Paper A", "Printing Room", "Distribution List"],
    "investigator_framing": (
      "One staff member may have leaked the exam papers prematurely. "
      "You don't know who."
    ),
    "evidence_hint": "print logs, access records, timestamps, communication logs",
    "investigator_a_title": "Principal",
    "investigator_b_title": "Academic Coordinator",
    "investigator_c_title": "Exam Controller"
  },

  "insurance_claim_fraud": {
    "name": "Insurance Claim Fraud",
    "flayer_role": "a claims adjuster reviewing submissions",
    "setting": "an insurance company. Multiple suspicious claims were approved without proper verification",
    "targets": ["Claim File A", "Approval System", "Verification Reports"],
    "investigator_framing": (
      "One adjuster may have knowingly approved fraudulent claims. "
      "You don't know who."
    ),
    "evidence_hint": "claim histories, approval timestamps, verification logs, payout records",
    "investigator_a_title": "Claims Manager",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Audit Director"
  },

  "retail_inventory_shrinkage": {
    "name": "Retail Inventory Shrinkage",
    "flayer_role": "a store manager overseeing stock control",
    "setting": "a retail chain. Inventory losses were detected across multiple product categories",
    "targets": ["Stock Room", "POS System", "Inventory Database"],
    "investigator_framing": (
      "One staff member may have been responsible for the unexplained inventory loss. "
      "You don't know who."
    ),
    "evidence_hint": "inventory audits, sales records, CCTV footage, stock adjustments",
    "investigator_a_title": "Regional Manager",
    "investigator_b_title": "Loss Prevention Officer",
    "investigator_c_title": "Store Auditor"
  },

  "telecom_network_outage": {
    "name": "Telecom Network Outage",
    "flayer_role": "a network engineer managing core systems",
    "setting": "a telecom provider. A large-scale network outage affected multiple regions",
    "targets": ["Core Network", "Routing System", "Switching Center"],
    "investigator_framing": (
      "One engineer may have caused or failed to prevent the outage. "
      "You don't know who."
    ),
    "evidence_hint": "network logs, configuration changes, outage reports, timestamps",
    "investigator_a_title": "Network Director",
    "investigator_b_title": "Systems Engineer",
    "investigator_c_title": "Operations Manager"
  },

  "agriculture_crop_failure": {
    "name": "Crop Failure Incident",
    "flayer_role": "an agronomist overseeing field operations",
    "setting": "a large agricultural farm. A major crop failed unexpectedly across several fields",
    "targets": ["Field A", "Irrigation System", "Fertilizer Supply"],
    "investigator_framing": (
      "One team member may have mismanaged or altered agricultural inputs. "
      "You don't know who."
    ),
    "evidence_hint": "soil reports, irrigation logs, fertilizer records, weather data",
    "investigator_a_title": "Farm Director",
    "investigator_b_title": "Agricultural Scientist",
    "investigator_c_title": "Operations Supervisor"
  },

  "space_mission_data_loss": {
    "name": "Space Mission Data Loss",
    "flayer_role": "a mission data engineer",
    "setting": "a space agency. Critical telemetry data was lost during transmission",
    "targets": ["Telemetry Stream", "Ground Station", "Data Archive"],
    "investigator_framing": (
      "One engineer may have caused or failed to prevent the data loss. "
      "You don't know who."
    ),
    "evidence_hint": "transmission logs, system diagnostics, timestamps, error reports",
    "investigator_a_title": "Mission Director",
    "investigator_b_title": "Systems Engineer",
    "investigator_c_title": "Data Analyst"
  },

  "charity_fund_misallocation": {
    "name": "Charity Fund Misallocation",
    "flayer_role": "a program coordinator managing fund distribution",
    "setting": "a non-profit organization. Donated funds were not used for intended programs",
    "targets": ["Program A", "Fund Ledger", "Distribution Records"],
    "investigator_framing": (
      "One coordinator may have misallocated charitable funds. "
      "You don't know who."
    ),
    "evidence_hint": "financial records, allocation logs, approval chains, timestamps",
    "investigator_a_title": "Executive Director",
    "investigator_b_title": "Financial Auditor",
    "investigator_c_title": "Program Manager"
  },

  "casino_chip_fraud": {
    "name": "Casino Chip Fraud",
    "flayer_role": "a floor supervisor overseeing table games",
    "setting": "a casino. High-value chips were found circulating without proper issuance",
    "targets": ["Chip Inventory", "Gaming Tables", "Cashier Desk"],
    "investigator_framing": (
      "One staff member may have introduced or allowed fraudulent chips into circulation. "
      "You don't know who."
    ),
    "evidence_hint": "chip tracking logs, surveillance footage, transaction records, timestamps",
    "investigator_a_title": "Casino Manager",
    "investigator_b_title": "Security Chief",
    "investigator_c_title": "Compliance Officer"
  },

  "restaurant_food_contamination": {
    "name": "Food Contamination Incident",
    "flayer_role": "a kitchen supervisor managing food preparation",
    "setting": "a restaurant chain. Several customers reported illness after dining",
    "targets": ["Kitchen Station", "Ingredient Supply", "Food Storage"],
    "investigator_framing": (
      "One staff member may have mishandled food safety procedures. "
      "You don't know who."
    ),
    "evidence_hint": "food safety logs, supplier records, kitchen schedules, inspection reports",
    "investigator_a_title": "Restaurant Manager",
    "investigator_b_title": "Health Inspector",
    "investigator_c_title": "Operations Head"
  },

  "shipping_port_delay": {
    "name": "Port Shipment Delay",
    "flayer_role": "a port operations officer managing cargo flow",
    "setting": "a shipping port. Critical cargo shipments were delayed without clear cause",
    "targets": ["Dock A", "Cargo Manifest", "Scheduling System"],
    "investigator_framing": (
      "One officer may have caused or failed to resolve the delay. "
      "You don't know who."
    ),
    "evidence_hint": "cargo logs, scheduling records, timestamps, communication logs",
    "investigator_a_title": "Port Director",
    "investigator_b_title": "Logistics Supervisor",
    "investigator_c_title": "Operations Analyst"
  },

  "cyber_fraud_payment_gateway": {
    "name": "Payment Gateway Fraud",
    "flayer_role": "a payment systems engineer",
    "setting": "a fintech platform. Unauthorized transactions were processed through the gateway",
    "targets": ["Gateway System", "Transaction Processor", "Fraud Detection Module"],
    "investigator_framing": (
      "One engineer may have introduced or failed to detect fraudulent transactions. "
      "You don't know who."
    ),
    "evidence_hint": "transaction logs, fraud alerts, system configs, timestamps",
    "investigator_a_title": "Fintech Director",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Security Engineer"
  },

  "museum_artwork_swap": {
    "name": "Artwork Substitution",
    "flayer_role": "a museum curator with collection access",
    "setting": "a museum. A valuable artwork was replaced with a replica without immediate detection",
    "targets": ["Exhibit Hall", "Storage Vault", "Inventory Records"],
    "investigator_framing": (
      "One staff member may have been involved in substituting the artwork. "
      "You don't know who."
    ),
    "evidence_hint": "inventory logs, surveillance footage, access records, handling reports",
    "investigator_a_title": "Museum Director",
    "investigator_b_title": "Art Conservator",
    "investigator_c_title": "Security Chief"
  },

  "university_grant_misuse": {
    "name": "Research Grant Misuse",
    "flayer_role": "a research administrator managing grant funds",
    "setting": "a university. Grant money was spent outside approved research purposes",
    "targets": ["Grant Account", "Expense Reports", "Project Budget"],
    "investigator_framing": (
      "One administrator may have misused research funding. "
      "You don't know who."
    ),
    "evidence_hint": "expense logs, approval records, timestamps, audit trails",
    "investigator_a_title": "Dean",
    "investigator_b_title": "Financial Auditor",
    "investigator_c_title": "Research Oversight Officer"
  },
    "rail_signal_failure": {
    "name": "Rail Signal Failure",
    "flayer_role": "a signaling engineer responsible for track systems",
    "setting": "a railway network. A signal malfunction caused two trains to nearly collide",
    "targets": ["Signal Node 12", "Control Panel", "Track Circuit Logs"],
    "investigator_framing": "One engineer may have misconfigured or failed to maintain the signaling system. You don't know who.",
    "evidence_hint": "signal logs, maintenance records, control inputs, timestamps",
    "investigator_a_title": "Rail Operations Chief",
    "investigator_b_title": "Safety Inspector",
    "investigator_c_title": "Systems Engineer"
  },

  "water_treatment_contamination": {
    "name": "Water Treatment Contamination",
    "flayer_role": "a plant technician overseeing filtration systems",
    "setting": "a municipal water facility. Contaminants were detected in treated water output",
    "targets": ["Filtration Unit", "Chemical Dosing System", "Reservoir Output"],
    "investigator_framing": "One technician may have mishandled or altered treatment processes. You don't know who.",
    "evidence_hint": "chemical logs, system readings, maintenance schedules, lab results",
    "investigator_a_title": "Plant Director",
    "investigator_b_title": "Environmental Auditor",
    "investigator_c_title": "Operations Supervisor"
  },

  "hotel_booking_manipulation": {
    "name": "Booking System Manipulation",
    "flayer_role": "a reservations manager with system access",
    "setting": "a hotel chain. Room bookings were altered, causing overbooking and losses",
    "targets": ["Reservation System", "Booking Records", "Payment Logs"],
    "investigator_framing": "One staff member may have manipulated booking records. You don't know who.",
    "evidence_hint": "booking edits, payment timestamps, user activity logs, cancellations",
    "investigator_a_title": "Hotel Director",
    "investigator_b_title": "Revenue Manager",
    "investigator_c_title": "Audit Officer"
  },

  "warehouse_robot_malfunction": {
    "name": "Warehouse Robot Malfunction",
    "flayer_role": "a robotics technician maintaining automation systems",
    "setting": "an automated warehouse. Robots began misrouting inventory items",
    "targets": ["Robot Fleet", "Routing Algorithm", "Inventory Grid"],
    "investigator_framing": "One technician may have introduced or failed to detect system faults. You don't know who.",
    "evidence_hint": "robot logs, pathing data, system updates, error reports",
    "investigator_a_title": "Automation Lead",
    "investigator_b_title": "Systems Engineer",
    "investigator_c_title": "Operations Manager"
  },

  "mining_safety_violation": {
    "name": "Mining Safety Violation",
    "flayer_role": "a site supervisor responsible for safety compliance",
    "setting": "a mining operation. Safety protocols were bypassed leading to an accident",
    "targets": ["Mine Shaft A", "Safety Checklist", "Equipment Logs"],
    "investigator_framing": "One supervisor may have ignored or altered safety procedures. You don't know who.",
    "evidence_hint": "inspection reports, equipment usage logs, shift records, incident timelines",
    "investigator_a_title": "Mine Director",
    "investigator_b_title": "Safety Officer",
    "investigator_c_title": "Operations Head"
  },

  "fashion_design_leak": {
    "name": "Fashion Design Leak",
    "flayer_role": "a design assistant with access to collections",
    "setting": "a fashion house. Upcoming collection designs appeared in a competitor's lineup",
    "targets": ["Design Portfolio", "Studio Files", "Prototype Samples"],
    "investigator_framing": "One team member may have leaked confidential designs. You don't know who.",
    "evidence_hint": "file access logs, export records, studio access times, communication logs",
    "investigator_a_title": "Creative Director",
    "investigator_b_title": "Brand Manager",
    "investigator_c_title": "Legal Advisor"
  },

  "music_streaming_data_tampering": {
    "name": "Streaming Data Tampering",
    "flayer_role": "a data analyst managing platform metrics",
    "setting": "a music streaming service. Play counts for certain artists were artificially inflated",
    "targets": ["Analytics Dashboard", "Play Count System", "User Activity Logs"],
    "investigator_framing": "One analyst may have manipulated streaming metrics. You don't know who.",
    "evidence_hint": "data anomalies, query logs, timestamps, user patterns",
    "investigator_a_title": "Product Head",
    "investigator_b_title": "Data Scientist",
    "investigator_c_title": "Audit Lead"
  },

  "construction_blueprint_error": {
    "name": "Blueprint Miscalculation",
    "flayer_role": "a civil engineer responsible for design approval",
    "setting": "a construction project. Structural inconsistencies were discovered mid-build",
    "targets": ["Blueprint A", "Structural Model", "Approval Documents"],
    "investigator_framing": "One engineer may have introduced or overlooked critical design errors. You don't know who.",
    "evidence_hint": "design revisions, approval timestamps, calculation sheets, inspection reports",
    "investigator_a_title": "Project Director",
    "investigator_b_title": "Structural Engineer",
    "investigator_c_title": "Quality Inspector"
  },

  "bank_atm_skimming": {
    "name": "ATM Skimming Incident",
    "flayer_role": "a maintenance contractor with ATM access",
    "setting": "a banking network. Customer accounts were compromised via ATM machines",
    "targets": ["ATM Unit A", "Card Reader", "Transaction Logs"],
    "investigator_framing": "One individual with access to ATMs may have installed skimming devices. You don't know who.",
    "evidence_hint": "maintenance logs, CCTV footage, transaction anomalies, access records",
    "investigator_a_title": "Security Manager",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Operations Head"
  },

  "hotel_food_supply_issue": {
    "name": "Food Supply Tampering",
    "flayer_role": "a procurement officer handling supplier contracts",
    "setting": "a hotel chain. Food supplies were found to be substandard or tampered",
    "targets": ["Supplier A", "Inventory Stock", "Delivery Records"],
    "investigator_framing": "One staff member may have approved compromised suppliers or deliveries. You don't know who.",
    "evidence_hint": "supplier logs, delivery timestamps, inspection reports, invoices",
    "investigator_a_title": "Procurement Head",
    "investigator_b_title": "Quality Inspector",
    "investigator_c_title": "Operations Manager"
  },

  "airport_baggage_misrouting": {
    "name": "Baggage Misrouting Incident",
    "flayer_role": "a baggage handling supervisor",
    "setting": "an airport. Multiple bags were rerouted to incorrect international destinations",
    "targets": ["Sorting System", "Conveyor Network", "Flight Allocation Logs"],
    "investigator_framing": "One staff member may have altered routing logic or mishandled baggage flows. You don't know who.",
    "evidence_hint": "routing logs, baggage scans, timestamps, shift records",
    "investigator_a_title": "Airport Operations Head",
    "investigator_b_title": "Logistics Manager",
    "investigator_c_title": "Security Officer"
  },

  "shipping_manifest_tampering": {
    "name": "Shipping Manifest Tampering",
    "flayer_role": "a freight documentation officer",
    "setting": "a shipping company. Cargo manifests were altered leading to missing goods",
    "targets": ["Manifest A", "Cargo Database", "Port Records"],
    "investigator_framing": "One officer may have modified shipment records. You don't know who.",
    "evidence_hint": "document revisions, cargo logs, timestamps, audit trails",
    "investigator_a_title": "Port Manager",
    "investigator_b_title": "Logistics Auditor",
    "investigator_c_title": "Operations Supervisor"
  },

  "tv_broadcast_interruption": {
    "name": "Broadcast Interruption",
    "flayer_role": "a broadcast engineer managing live feeds",
    "setting": "a television network. A live broadcast was abruptly replaced with unauthorized content",
    "targets": ["Live Feed", "Control Room", "Transmission System"],
    "investigator_framing": "One engineer may have interfered with the broadcast system. You don't know who.",
    "evidence_hint": "transmission logs, control inputs, access records, timestamps",
    "investigator_a_title": "Broadcast Director",
    "investigator_b_title": "Technical Lead",
    "investigator_c_title": "Network Manager"
  },

  "library_archive_loss": {
    "name": "Archive Record Loss",
    "flayer_role": "an archivist managing historical records",
    "setting": "a national library. Rare documents went missing from the archive",
    "targets": ["Archive Vault", "Catalog System", "Loan Records"],
    "investigator_framing": "One staff member may have mishandled or removed archived materials. You don't know who.",
    "evidence_hint": "checkout logs, access records, surveillance timestamps, catalog updates",
    "investigator_a_title": "Library Director",
    "investigator_b_title": "Records Manager",
    "investigator_c_title": "Security Supervisor"
  },

  "theme_park_ride_malfunction": {
    "name": "Ride Malfunction Incident",
    "flayer_role": "a ride maintenance technician",
    "setting": "a theme park. A major ride stopped abruptly during operation",
    "targets": ["Ride System", "Control Panel", "Maintenance Logs"],
    "investigator_framing": "One technician may have failed to properly maintain or inspect the ride. You don't know who.",
    "evidence_hint": "maintenance records, system diagnostics, error logs, shift schedules",
    "investigator_a_title": "Park Manager",
    "investigator_b_title": "Safety Inspector",
    "investigator_c_title": "Engineering Lead"
  },

  "publishing_manuscript_theft": {
    "name": "Manuscript Theft",
    "flayer_role": "an editorial assistant with manuscript access",
    "setting": "a publishing house. An unpublished manuscript appeared online",
    "targets": ["Manuscript File", "Editorial System", "Author Submissions"],
    "investigator_framing": "One staff member may have leaked or stolen the manuscript. You don't know who.",
    "evidence_hint": "file access logs, download history, timestamps, email records",
    "investigator_a_title": "Editor-in-Chief",
    "investigator_b_title": "Legal Advisor",
    "investigator_c_title": "Publishing Manager"
  },

  "pharma_trial_blind_break": {
    "name": "Trial Blind Break",
    "flayer_role": "a clinical data manager",
    "setting": "a pharmaceutical trial. The blind was broken before completion of the study",
    "targets": ["Trial Data", "Patient Allocation", "Blind Key"],
    "investigator_framing": "One team member may have accessed restricted trial information prematurely. You don't know who.",
    "evidence_hint": "data access logs, timestamps, authorization records, system alerts",
    "investigator_a_title": "Trial Director",
    "investigator_b_title": "Clinical Auditor",
    "investigator_c_title": "Regulatory Officer"
  },

  "insurance_policy_backdating": {
    "name": "Policy Backdating",
    "flayer_role": "an underwriting officer",
    "setting": "an insurance firm. Policies were backdated to cover prior incidents",
    "targets": ["Policy Records", "Claims Files", "Approval Logs"],
    "investigator_framing": "One officer may have manipulated policy issuance dates. You don't know who.",
    "evidence_hint": "policy timestamps, approval chains, claim records, audit logs",
    "investigator_a_title": "Underwriting Manager",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Audit Director"
  },

  "university_admission_irregularity": {
    "name": "Admission Irregularity",
    "flayer_role": "an admissions officer reviewing applications",
    "setting": "a university. Several applicants were admitted despite failing eligibility criteria",
    "targets": ["Application Files", "Admission Portal", "Evaluation Records"],
    "investigator_framing": "One staff member may have manipulated admission decisions. You don't know who.",
    "evidence_hint": "application scores, decision logs, timestamps, reviewer notes",
    "investigator_a_title": "Admissions Director",
    "investigator_b_title": "Academic Auditor",
    "investigator_c_title": "Registrar"
  },

  "energy_meter_tampering": {
    "name": "Energy Meter Tampering",
    "flayer_role": "a field technician installing meters",
    "setting": "an energy utility. Multiple smart meters reported inconsistent readings",
    "targets": ["Meter Network", "Installation Records", "Billing System"],
    "investigator_framing": "One technician may have tampered with meter configurations. You don't know who.",
    "evidence_hint": "meter logs, installation timestamps, billing anomalies, device configs",
    "investigator_a_title": "Utility Manager",
    "investigator_b_title": "Technical Auditor",
    "investigator_c_title": "Field Supervisor"
  },
    "marine_navigation_error": {
    "name": "Marine Navigation Error",
    "flayer_role": "a navigation officer overseeing vessel routing",
    "setting": "a cargo ship at sea. The vessel deviated significantly from its planned route",
    "targets": ["Navigation System", "Route Plan", "Bridge Controls"],
    "investigator_framing": "One officer may have altered or mismanaged navigation instructions. You don't know who.",
    "evidence_hint": "GPS logs, route deviations, control inputs, timestamps",
    "investigator_a_title": "Ship Captain",
    "investigator_b_title": "Fleet Manager",
    "investigator_c_title": "Maritime Safety Officer"
  },

  "fire_department_response_delay": {
    "name": "Emergency Response Delay",
    "flayer_role": "a dispatch operator coordinating emergency calls",
    "setting": "a city fire department. Emergency response was delayed during a critical incident",
    "targets": ["Dispatch System", "Call Logs", "Response Units"],
    "investigator_framing": "One operator may have mishandled or delayed emergency dispatch. You don't know who.",
    "evidence_hint": "call timestamps, dispatch records, unit logs, communication transcripts",
    "investigator_a_title": "Fire Chief",
    "investigator_b_title": "Operations Officer",
    "investigator_c_title": "Internal Auditor"
  },

  "zoo_animal_escape": {
    "name": "Animal Escape Incident",
    "flayer_role": "a zookeeper responsible for enclosure maintenance",
    "setting": "a city zoo. A dangerous animal escaped its enclosure",
    "targets": ["Enclosure A", "Security Gate", "Feeding Schedule"],
    "investigator_framing": "One staff member may have failed to secure the enclosure properly. You don't know who.",
    "evidence_hint": "lock logs, surveillance footage, maintenance records, shift timings",
    "investigator_a_title": "Zoo Director",
    "investigator_b_title": "Wildlife Officer",
    "investigator_c_title": "Safety Inspector"
  },

  "printing_press_error": {
    "name": "Printing Error Incident",
    "flayer_role": "a press operator managing print runs",
    "setting": "a printing facility. Thousands of copies were printed with incorrect content",
    "targets": ["Print Batch A", "Layout File", "Press Machine"],
    "investigator_framing": "One operator may have used incorrect files or settings. You don't know who.",
    "evidence_hint": "print logs, file versions, timestamps, machine settings",
    "investigator_a_title": "Production Manager",
    "investigator_b_title": "Quality Controller",
    "investigator_c_title": "Operations Supervisor"
  },

  "theme_park_ticket_fraud": {
    "name": "Ticketing Fraud",
    "flayer_role": "a ticketing supervisor with system access",
    "setting": "a theme park. Unauthorized tickets were issued and used for entry",
    "targets": ["Ticketing System", "Entry Gates", "Sales Records"],
    "investigator_framing": "One staff member may have generated fraudulent tickets. You don't know who.",
    "evidence_hint": "ticket logs, entry scans, payment records, timestamps",
    "investigator_a_title": "Park Director",
    "investigator_b_title": "Finance Officer",
    "investigator_c_title": "Security Manager"
  },

  "meteorological_data_error": {
    "name": "Weather Data Error",
    "flayer_role": "a meteorologist managing forecasting models",
    "setting": "a weather agency. Incorrect forecasts were issued for a major storm",
    "targets": ["Forecast Model", "Data Feed", "Alert System"],
    "investigator_framing": "One analyst may have misinterpreted or altered weather data. You don't know who.",
    "evidence_hint": "model outputs, data inputs, timestamps, forecast logs",
    "investigator_a_title": "Chief Meteorologist",
    "investigator_b_title": "Data Analyst",
    "investigator_c_title": "Operations Lead"
  },

  "waste_management_misrouting": {
    "name": "Waste Misrouting Incident",
    "flayer_role": "a logistics coordinator handling waste disposal routes",
    "setting": "a waste management company. Hazardous waste was sent to the wrong facility",
    "targets": ["Route Plan", "Disposal Site", "Transport Logs"],
    "investigator_framing": "One coordinator may have misrouted hazardous materials. You don't know who.",
    "evidence_hint": "routing logs, transport records, timestamps, facility logs",
    "investigator_a_title": "Operations Manager",
    "investigator_b_title": "Environmental Officer",
    "investigator_c_title": "Compliance Auditor"
  },

  "auction_bid_manipulation": {
    "name": "Auction Bid Manipulation",
    "flayer_role": "an auction coordinator overseeing bids",
    "setting": "an auction house. Bid values were altered during a high-value auction",
    "targets": ["Bid Records", "Auction System", "Lot A"],
    "investigator_framing": "One staff member may have manipulated bidding data. You don't know who.",
    "evidence_hint": "bid logs, timestamps, user activity, transaction records",
    "investigator_a_title": "Auction Director",
    "investigator_b_title": "Finance Auditor",
    "investigator_c_title": "Legal Advisor"
  },

  "gym_membership_fraud": {
    "name": "Membership Fraud",
    "flayer_role": "a front desk manager handling registrations",
    "setting": "a fitness chain. Fake memberships were created and used",
    "targets": ["Membership System", "Payment Records", "Access Logs"],
    "investigator_framing": "One staff member may have created fraudulent accounts. You don't know who.",
    "evidence_hint": "registration logs, payment data, entry scans, timestamps",
    "investigator_a_title": "Branch Manager",
    "investigator_b_title": "Finance Officer",
    "investigator_c_title": "Operations Supervisor"
  },

  "library_system_hack": {
    "name": "Library System Breach",
    "flayer_role": "an IT administrator maintaining catalog systems",
    "setting": "a public library network. User records were accessed and altered",
    "targets": ["Catalog System", "User Database", "Access Logs"],
    "investigator_framing": "One staff member may have misused system access. You don't know who.",
    "evidence_hint": "login records, database queries, timestamps, activity logs",
    "investigator_a_title": "IT Director",
    "investigator_b_title": "Security Analyst",
    "investigator_c_title": "Library Administrator"
  },

  "power_plant_shutdown": {
    "name": "Unexpected Plant Shutdown",
    "flayer_role": "a control room operator managing power output",
    "setting": "a power plant. The facility shut down unexpectedly during peak load",
    "targets": ["Control Panel", "Reactor Unit", "Monitoring System"],
    "investigator_framing": "One operator may have triggered or failed to prevent the shutdown. You don't know who.",
    "evidence_hint": "control logs, system alerts, timestamps, sensor data",
    "investigator_a_title": "Plant Director",
    "investigator_b_title": "Systems Engineer",
    "investigator_c_title": "Operations Analyst"
  },

  "call_center_data_leak": {
    "name": "Customer Data Leak",
    "flayer_role": "a call center supervisor with database access",
    "setting": "a customer support center. Client information was leaked externally",
    "targets": ["Customer Database", "CRM System", "Call Logs"],
    "investigator_framing": "One employee may have exposed sensitive customer data. You don't know who.",
    "evidence_hint": "access logs, call recordings, export activity, timestamps",
    "investigator_a_title": "Operations Head",
    "investigator_b_title": "Data Protection Officer",
    "investigator_c_title": "Audit Manager"
  },

  "pharmacy_prescription_forgery": {
    "name": "Prescription Forgery",
    "flayer_role": "a pharmacy assistant handling prescriptions",
    "setting": "a retail pharmacy. Forged prescriptions were processed successfully",
    "targets": ["Prescription Records", "Dispensing System", "Inventory Logs"],
    "investigator_framing": "One staff member may have approved forged prescriptions. You don't know who.",
    "evidence_hint": "prescription logs, approval timestamps, stock records, patient IDs",
    "investigator_a_title": "Chief Pharmacist",
    "investigator_b_title": "Medical Auditor",
    "investigator_c_title": "Store Manager"
  },

  "warehouse_temperature_failure": {
    "name": "Cold Storage Failure",
    "flayer_role": "a facility technician managing storage systems",
    "setting": "a cold storage warehouse. Temperature controls failed, damaging goods",
    "targets": ["Cooling System", "Storage Unit", "Sensor Logs"],
    "investigator_framing": "One technician may have failed to maintain or monitor the system. You don't know who.",
    "evidence_hint": "temperature logs, maintenance records, sensor alerts, timestamps",
    "investigator_a_title": "Facility Manager",
    "investigator_b_title": "Quality Inspector",
    "investigator_c_title": "Operations Lead"
  },

  "film_editing_tamper": {
    "name": "Film Edit Tampering",
    "flayer_role": "a video editor with access to final cuts",
    "setting": "a post-production studio. Final footage was altered before release",
    "targets": ["Final Cut", "Editing System", "Backup Files"],
    "investigator_framing": "One editor may have tampered with the final version. You don't know who.",
    "evidence_hint": "edit logs, file versions, timestamps, render history",
    "investigator_a_title": "Director",
    "investigator_b_title": "Producer",
    "investigator_c_title": "Post-Production Lead"
  },

  "restaurant_billing_error": {
    "name": "Billing System Error",
    "flayer_role": "a cashier managing transactions",
    "setting": "a restaurant chain. Bills were incorrectly calculated for multiple customers",
    "targets": ["POS System", "Billing Records", "Payment Logs"],
    "investigator_framing": "One staff member may have altered billing entries. You don't know who.",
    "evidence_hint": "transaction logs, receipts, timestamps, system records",
    "investigator_a_title": "Restaurant Manager",
    "investigator_b_title": "Finance Auditor",
    "investigator_c_title": "Operations Supervisor"
  },

  "delivery_route_fraud": {
    "name": "Delivery Route Fraud",
    "flayer_role": "a delivery coordinator assigning routes",
    "setting": "a logistics company. Drivers reported incorrect delivery routes and missing packages",
    "targets": ["Route Assignments", "Delivery Logs", "Tracking System"],
    "investigator_framing": "One coordinator may have manipulated route assignments. You don't know who.",
    "evidence_hint": "GPS logs, delivery timestamps, assignment records, package scans",
    "investigator_a_title": "Logistics Manager",
    "investigator_b_title": "Fleet Supervisor",
    "investigator_c_title": "Audit Officer"
  },

  "hotel_security_camera_failure": {
    "name": "Security Camera Failure",
    "flayer_role": "a security technician maintaining surveillance systems",
    "setting": "a hotel. Surveillance cameras failed during a critical incident",
    "targets": ["Camera Network", "Control Room", "Recording System"],
    "investigator_framing": "One technician may have disabled or failed to maintain surveillance systems. You don't know who.",
    "evidence_hint": "camera logs, system alerts, maintenance records, timestamps",
    "investigator_a_title": "Security Head",
    "investigator_b_title": "IT Manager",
    "investigator_c_title": "Operations Director"
  },

  "bank_loan_approval_bias": {
    "name": "Loan Approval Irregularity",
    "flayer_role": "a loan officer reviewing applications",
    "setting": "a bank. Several high-risk loans were approved against policy",
    "targets": ["Loan Applications", "Approval System", "Risk Reports"],
    "investigator_framing": "One officer may have bypassed risk protocols. You don't know who.",
    "evidence_hint": "approval logs, risk scores, timestamps, audit trails",
    "investigator_a_title": "Branch Manager",
    "investigator_b_title": "Risk Analyst",
    "investigator_c_title": "Compliance Officer"
  },

  "school_transport_mismanagement": {
    "name": "School Transport Mismanagement",
    "flayer_role": "a transport coordinator managing school buses",
    "setting": "a school system. Several buses deviated from assigned routes",
    "targets": ["Bus Routes", "Transport Logs", "GPS System"],
    "investigator_framing": "One coordinator may have mismanaged or altered routes. You don't know who.",
    "evidence_hint": "GPS data, route logs, timestamps, driver reports",
    "investigator_a_title": "School Administrator",
    "investigator_b_title": "Transport Manager",
    "investigator_c_title": "Operations Officer"
  },

  "warehouse_label_swap": {
    "name": "Inventory Label Swap",
    "flayer_role": "a warehouse supervisor managing stock labeling",
    "setting": "a distribution center. Product labels were swapped leading to shipment errors",
    "targets": ["Labeling System", "Inventory Records", "Dispatch Logs"],
    "investigator_framing": "One staff member may have altered or misapplied labels. You don't know who.",
    "evidence_hint": "label logs, inventory scans, timestamps, dispatch records",
    "investigator_a_title": "Warehouse Manager",
    "investigator_b_title": "Quality Inspector",
    "investigator_c_title": "Operations Lead"
  },

  "airport_ground_ops_error": {
    "name": "Ground Operations Error",
    "flayer_role": "a ground operations coordinator",
    "setting": "an airport. Aircraft servicing was improperly handled before departure",
    "targets": ["Service Logs", "Aircraft A", "Ground Equipment"],
    "investigator_framing": "One coordinator may have mishandled pre-flight operations. You don't know who.",
    "evidence_hint": "service records, timestamps, equipment logs, crew reports",
    "investigator_a_title": "Operations Director",
    "investigator_b_title": "Safety Inspector",
    "investigator_c_title": "Ground Manager"
  },

  "hospital_bed_allocation_issue": {
    "name": "Bed Allocation Issue",
    "flayer_role": "a hospital administrator managing patient assignments",
    "setting": "a hospital. Critical patients were assigned incorrect wards",
    "targets": ["Bed Allocation System", "Patient Records", "Ward Logs"],
    "investigator_framing": "One administrator may have mismanaged patient allocation. You don't know who.",
    "evidence_hint": "assignment logs, timestamps, patient IDs, system records",
    "investigator_a_title": "Hospital Director",
    "investigator_b_title": "Medical Auditor",
    "investigator_c_title": "Operations Manager"
  },

  "factory_quality_control_failure": {
    "name": "Quality Control Failure",
    "flayer_role": "a quality inspector responsible for final checks",
    "setting": "a manufacturing plant. Defective products passed quality inspection",
    "targets": ["Inspection Reports", "Production Line", "Batch Records"],
    "investigator_framing": "One inspector may have overlooked or bypassed quality checks. You don't know who.",
    "evidence_hint": "inspection logs, defect reports, timestamps, batch data",
    "investigator_a_title": "Plant Manager",
    "investigator_b_title": "Quality Lead",
    "investigator_c_title": "Operations Supervisor"
  },

  "university_grade_tampering": {
    "name": "Grade Tampering Incident",
    "flayer_role": "an academic staff member with grading access",
    "setting": "a university. Student grades were altered after submission deadlines",
    "targets": ["Grade Database", "Submission Portal", "Audit Logs"],
    "investigator_framing": "One staff member may have modified grades improperly. You don't know who.",
    "evidence_hint": "grade changes, timestamps, login records, audit trails",
    "investigator_a_title": "Dean",
    "investigator_b_title": "Registrar",
    "investigator_c_title": "Academic Auditor"
  },

  "retail_discount_abuse": {
    "name": "Discount Abuse Incident",
    "flayer_role": "a sales associate with discount privileges",
    "setting": "a retail store. Large unauthorized discounts were applied to purchases",
    "targets": ["POS System", "Discount Logs", "Sales Records"],
    "investigator_framing": "One employee may have misused discount privileges. You don't know who.",
    "evidence_hint": "discount logs, transaction records, timestamps, receipts",
    "investigator_a_title": "Store Manager",
    "investigator_b_title": "Finance Auditor",
    "investigator_c_title": "Operations Head"
  },

  "telecom_sim_swap": {
    "name": "SIM Swap Incident",
    "flayer_role": "a telecom support agent handling SIM requests",
    "setting": "a telecom provider. Multiple users reported unauthorized SIM swaps",
    "targets": ["SIM Database", "Customer Accounts", "Request Logs"],
    "investigator_framing": "One agent may have processed fraudulent SIM swap requests. You don't know who.",
    "evidence_hint": "request logs, authentication records, timestamps, account changes",
    "investigator_a_title": "Support Manager",
    "investigator_b_title": "Fraud Analyst",
    "investigator_c_title": "Security Officer"
  },

  "agriculture_supply_mixup": {
    "name": "Seed Supply Mix-Up",
    "flayer_role": "a supply manager distributing seeds",
    "setting": "an agricultural supplier. Farmers received incorrect seed varieties",
    "targets": ["Seed Inventory", "Distribution Records", "Packaging Unit"],
    "investigator_framing": "One staff member may have mismanaged or mislabeled supplies. You don't know who.",
    "evidence_hint": "inventory logs, packaging records, shipment data, timestamps",
    "investigator_a_title": "Supply Director",
    "investigator_b_title": "Agriculture Officer",
    "investigator_c_title": "Quality Inspector"
  },

  "space_equipment_fault": {
    "name": "Equipment Fault in Orbit",
    "flayer_role": "a systems engineer responsible for onboard equipment",
    "setting": "a space mission. Critical onboard equipment malfunctioned unexpectedly",
    "targets": ["Subsystem A", "Control Software", "Telemetry Logs"],
    "investigator_framing": "One engineer may have misconfigured or failed to validate system components. You don't know who.",
    "evidence_hint": "telemetry data, system logs, update history, timestamps",
    "investigator_a_title": "Mission Director",
    "investigator_b_title": "Systems Analyst",
    "investigator_c_title": "Engineering Lead"
  },

  "nonprofit_donor_data_misuse": {
    "name": "Donor Data Misuse",
    "flayer_role": "a fundraising coordinator with donor access",
    "setting": "a non-profit organization. Donor data was used for unauthorized campaigns",
    "targets": ["Donor Database", "Campaign System", "Email Logs"],
    "investigator_framing": "One coordinator may have misused donor information. You don't know who.",
    "evidence_hint": "email logs, database queries, campaign records, timestamps",
    "investigator_a_title": "Executive Director",
    "investigator_b_title": "Compliance Officer",
    "investigator_c_title": "Audit Manager"
  },
   "metro_card_system_failure": {
    "name": "Metro Card System Failure",
    "flayer_role": "a systems engineer maintaining fare infrastructure",
    "setting": "a city metro network. Commuters were unable to access stations due to card validation failures",
    "targets": ["Validation Gates", "Fare System", "Card Database"],
    "investigator_framing": "One engineer may have misconfigured or disrupted the fare system. You don't know who.",
    "evidence_hint": "system logs, validation errors, deployment timestamps, access records",
    "investigator_a_title": "Transit Director",
    "investigator_b_title": "Systems Analyst",
    "investigator_c_title": "Operations Manager"
  },

  "brewery_batch_spoilage": {
    "name": "Batch Spoilage Incident",
    "flayer_role": "a brewmaster overseeing fermentation",
    "setting": "a brewery. Multiple batches were spoiled before distribution",
    "targets": ["Fermentation Tank A", "Ingredient Supply", "Quality Logs"],
    "investigator_framing": "One staff member may have mishandled the brewing process. You don't know who.",
    "evidence_hint": "temperature logs, ingredient records, fermentation data, timestamps",
    "investigator_a_title": "Production Head",
    "investigator_b_title": "Quality Inspector",
    "investigator_c_title": "Operations Supervisor"
  },

  "airport_runway_allocation_error": {
    "name": "Runway Allocation Error",
    "flayer_role": "an air traffic coordinator managing runway assignments",
    "setting": "an airport. Two incoming aircraft were assigned the same runway slot",
    "targets": ["Runway Schedule", "Control Tower System", "Flight Logs"],
    "investigator_framing": "One coordinator may have misassigned runway slots. You don't know who.",
    "evidence_hint": "assignment logs, communication records, timestamps, radar data",
    "investigator_a_title": "Air Traffic Chief",
    "investigator_b_title": "Safety Officer",
    "investigator_c_title": "Operations Director"
  },

  "bookstore_inventory_fraud": {
    "name": "Inventory Fraud",
    "flayer_role": "a store supervisor managing stock",
    "setting": "a bookstore chain. High-value books were missing from inventory",
    "targets": ["Stock Records", "POS System", "Storage Room"],
    "investigator_framing": "One staff member may have manipulated inventory records. You don't know who.",
    "evidence_hint": "inventory logs, sales records, CCTV footage, timestamps",
    "investigator_a_title": "Store Manager",
    "investigator_b_title": "Audit Officer",
    "investigator_c_title": "Operations Head"
  },

  "courier_package_switch": {
    "name": "Package Switch Incident",
    "flayer_role": "a sorting facility supervisor",
    "setting": "a courier company. Packages were delivered to incorrect recipients",
    "targets": ["Sorting Line", "Labeling System", "Dispatch Logs"],
    "investigator_framing": "One staff member may have swapped or mislabeled packages. You don't know who.",
    "evidence_hint": "sorting logs, label scans, timestamps, route records",
    "investigator_a_title": "Logistics Manager",
    "investigator_b_title": "Quality Inspector",
    "investigator_c_title": "Dispatch Supervisor"
  },

  "cinema_projection_error": {
    "name": "Projection System Error",
    "flayer_role": "a projection technician managing screenings",
    "setting": "a cinema. Incorrect film content was played during scheduled screenings",
    "targets": ["Projection System", "Media Server", "Schedule Logs"],
    "investigator_framing": "One technician may have loaded or configured incorrect media. You don't know who.",
    "evidence_hint": "playback logs, file access, scheduling data, timestamps",
    "investigator_a_title": "Theater Manager",
    "investigator_b_title": "Technical Lead",
    "investigator_c_title": "Operations Supervisor"
  },

  "fisheries_stock_discrepancy": {
    "name": "Fish Stock Discrepancy",
    "flayer_role": "a fisheries officer managing catch records",
    "setting": "a fisheries department. Reported fish stock levels did not match actual counts",
    "targets": ["Catch Logs", "Storage Facility", "Distribution Records"],
    "investigator_framing": "One officer may have altered stock reporting. You don't know who.",
    "evidence_hint": "logbooks, inventory counts, timestamps, shipment data",
    "investigator_a_title": "Fisheries Director",
    "investigator_b_title": "Audit Officer",
    "investigator_c_title": "Operations Manager"
  },

  "event_ticket_overissue": {
    "name": "Ticket Over-Issuance",
    "flayer_role": "an event coordinator handling ticket distribution",
    "setting": "a concert venue. More tickets were issued than available seats",
    "targets": ["Ticketing System", "Entry Gates", "Sales Records"],
    "investigator_framing": "One coordinator may have overissued or duplicated tickets. You don't know who.",
    "evidence_hint": "ticket logs, entry scans, timestamps, payment records",
    "investigator_a_title": "Event Director",
    "investigator_b_title": "Finance Manager",
    "investigator_c_title": "Security Head"
  },

  "car_rental_return_fraud": {
    "name": "Vehicle Return Fraud",
    "flayer_role": "a rental agent handling returns",
    "setting": "a car rental company. Vehicles were marked returned but remained missing",
    "targets": ["Return Records", "Vehicle Tracking", "Rental Logs"],
    "investigator_framing": "One agent may have falsified return entries. You don't know who.",
    "evidence_hint": "GPS data, return timestamps, inspection logs, contracts",
    "investigator_a_title": "Branch Manager",
    "investigator_b_title": "Fleet Supervisor",
    "investigator_c_title": "Audit Officer"
  },

  "factory_power_usage_anomaly": {
    "name": "Power Usage Anomaly",
    "flayer_role": "an energy manager monitoring consumption",
    "setting": "a manufacturing plant. Energy usage spiked without corresponding production output",
    "targets": ["Power Meters", "Production Line", "Energy Logs"],
    "investigator_framing": "One staff member may have misused or misreported energy consumption. You don't know who.",
    "evidence_hint": "meter readings, production data, timestamps, usage reports",
    "investigator_a_title": "Plant Director",
    "investigator_b_title": "Energy Auditor",
    "investigator_c_title": "Operations Lead"
  },

  "news_ad_revenue_mismatch": {
    "name": "Ad Revenue Discrepancy",
    "flayer_role": "a revenue analyst tracking ad performance",
    "setting": "a digital media company. Reported ad revenue did not match actual earnings",
    "targets": ["Ad Platform", "Revenue Reports", "Analytics Dashboard"],
    "investigator_framing": "One analyst may have altered revenue data. You don't know who.",
    "evidence_hint": "analytics logs, revenue records, timestamps, report exports",
    "investigator_a_title": "Finance Director",
    "investigator_b_title": "Data Analyst",
    "investigator_c_title": "Audit Manager"
  },

  "pharmacy_expiry_mislabel": {
    "name": "Expiry Mislabeling",
    "flayer_role": "a pharmacy inventory manager",
    "setting": "a pharmacy. Medicines with incorrect expiry dates were dispensed",
    "targets": ["Inventory Labels", "Dispensing Records", "Stock Database"],
    "investigator_framing": "One staff member may have mislabeled product expiry dates. You don't know who.",
    "evidence_hint": "label logs, stock entries, timestamps, dispensing records",
    "investigator_a_title": "Chief Pharmacist",
    "investigator_b_title": "Quality Auditor",
    "investigator_c_title": "Store Manager"
  },

  "hotel_loyalty_points_fraud": {
    "name": "Loyalty Points Fraud",
    "flayer_role": "a rewards program manager",
    "setting": "a hotel chain. Loyalty points were credited without valid stays",
    "targets": ["Rewards System", "Customer Accounts", "Transaction Logs"],
    "investigator_framing": "One staff member may have manipulated loyalty accounts. You don't know who.",
    "evidence_hint": "account logs, transaction records, timestamps, audit trails",
    "investigator_a_title": "Marketing Director",
    "investigator_b_title": "Finance Auditor",
    "investigator_c_title": "Operations Head"
  },

  "school_attendance_falsification": {
    "name": "Attendance Falsification",
    "flayer_role": "a school administrator managing attendance records",
    "setting": "a school. Attendance records were altered for multiple students",
    "targets": ["Attendance System", "Class Logs", "Student Records"],
    "investigator_framing": "One staff member may have falsified attendance data. You don't know who.",
    "evidence_hint": "attendance logs, timestamps, login records, audit trails",
    "investigator_a_title": "Principal",
    "investigator_b_title": "Academic Coordinator",
    "investigator_c_title": "Audit Officer"
  },

  "telecom_billing_error": {
    "name": "Billing Calculation Error",
    "flayer_role": "a billing systems analyst",
    "setting": "a telecom company. Customers were overcharged due to incorrect billing calculations",
    "targets": ["Billing Engine", "Usage Records", "Customer Accounts"],
    "investigator_framing": "One analyst may have misconfigured billing rules. You don't know who.",
    "evidence_hint": "billing logs, usage data, timestamps, config changes",
    "investigator_a_title": "Finance Head",
    "investigator_b_title": "Systems Analyst",
    "investigator_c_title": "Audit Manager"
  },

  "construction_material_theft": {
    "name": "Material Theft Incident",
    "flayer_role": "a site supervisor managing construction materials",
    "setting": "a construction site. High-value materials went missing",
    "targets": ["Material Storage", "Inventory Logs", "Delivery Records"],
    "investigator_framing": "One staff member may have diverted materials. You don't know who.",
    "evidence_hint": "inventory records, delivery logs, CCTV footage, timestamps",
    "investigator_a_title": "Project Manager",
    "investigator_b_title": "Site Auditor",
    "investigator_c_title": "Operations Supervisor"
  },

  "food_delivery_app_glitch": {
    "name": "Order Processing Glitch",
    "flayer_role": "a backend developer managing order systems",
    "setting": "a food delivery platform. Orders were duplicated and misrouted",
    "targets": ["Order System", "Delivery Routing", "Restaurant Dashboard"],
    "investigator_framing": "One engineer may have introduced a system bug. You don't know who.",
    "evidence_hint": "system logs, order timestamps, routing data, deployment history",
    "investigator_a_title": "Product Manager",
    "investigator_b_title": "Engineering Lead",
    "investigator_c_title": "Operations Analyst"
  },

  "sports_ticket_scalping": {
    "name": "Ticket Scalping Ring",
    "flayer_role": "a ticketing official with system privileges",
    "setting": "a sports stadium. Tickets were resold illegally at inflated prices",
    "targets": ["Ticket Inventory", "Sales Platform", "Access Logs"],
    "investigator_framing": "One official may have facilitated ticket scalping. You don't know who.",
    "evidence_hint": "sales logs, account activity, timestamps, transaction data",
    "investigator_a_title": "Stadium Manager",
    "investigator_b_title": "Security Head",
    "investigator_c_title": "Finance Officer"
  },

  "museum_climate_control_failure": {
    "name": "Climate Control Failure",
    "flayer_role": "a facility engineer maintaining environmental systems",
    "setting": "a museum. Sensitive artifacts were damaged due to improper climate conditions",
    "targets": ["Climate System", "Exhibit Hall", "Sensor Logs"],
    "investigator_framing": "One engineer may have failed to maintain proper environmental controls. You don't know who.",
    "evidence_hint": "temperature logs, humidity data, maintenance records, timestamps",
    "investigator_a_title": "Museum Director",
    "investigator_b_title": "Conservation Expert",
    "investigator_c_title": "Facility Manager"
  },

  "bank_cheque_processing_error": {
    "name": "Cheque Processing Error",
    "flayer_role": "a clearing officer handling cheque verification",
    "setting": "a bank. Several cheques were cleared without proper validation",
    "targets": ["Clearing System", "Cheque Records", "Verification Logs"],
    "investigator_framing": "One officer may have bypassed verification procedures. You don't know who.",
    "evidence_hint": "processing logs, timestamps, approval records, audit trails",
    "investigator_a_title": "Branch Manager",
    "investigator_b_title": "Compliance Officer",
    "investigator_c_title": "Audit Lead"
  }
}

DEFAULT_SCENARIO = "corporate"

ALL_TARGET_NAMES: set[str] = {
    t.lower()
    for cfg in SCENARIO_CONFIGS.values()
    for t in cfg["targets"]
}
