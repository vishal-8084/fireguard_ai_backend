// sendMail.js
const nodemailer = require("nodemailer");
const path = require("path");

async function sendEmail(status, severity, timestamp, imagePath) {
  let transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: "infomotivefact@gmail.com",
      pass: "qlnl dqub rlfx tpsa"  
    }
  });

  let message = `
🔥 FIRE / SMOKE ALERT

Status   : ${status}
Severity : ${severity}
Time     : ${timestamp}

The first snapshot of the incident is attached for proof.
Please take necessary action.
`;

  // Base options
  let mailOptions = {
    from: '"Fire Alert System" <infomotivefact@gmail.com>',
    to: "vkngaming8085@gmail.com",
    subject: `🔥 Fire Alert | Status: ${status} | Severity: ${severity}`,
    text: message,
    attachments: []
  };

  // Agar image path milta hai toh use attach karein
  if (imagePath) {
    mailOptions.attachments.push({
      filename: path.basename(imagePath),
      path: imagePath 
    });
  }

  try {
    let info = await transporter.sendMail(mailOptions);
    console.log("📧 Email sent:", info.response);
  } catch (error) {
    console.log("❌ Error sending email:", error);
  }
}

// Arguments: status, severity, timestamp, imagePath
sendEmail(process.argv[2], process.argv[3], process.argv[4], process.argv[5]);