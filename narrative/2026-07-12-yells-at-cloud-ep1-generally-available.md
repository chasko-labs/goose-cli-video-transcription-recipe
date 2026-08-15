# Generally Available | Yells at Cloud Ep. 01

**Published:** July 11, 2026 | **Duration:** 1h 4m | [YouTube](https://www.youtube.com/watch?v=lWUu3V6dxpc) | [Spotify](https://open.spotify.com/episode/6FxW0YCX4CX7eNpBAtqbET) | [Apple Podcasts](https://podcasts.apple.com/us/podcast/yells-at-cloud/id6789969050) | [yellsatcloudpod.com](https://yellsatcloudpod.com/episodes/ep-1)

![The panel — Gunnar Grosch, AJ Stuyvenberg, Danielle Heberling, Matthew Bonig, Chris Williams](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/group-opening.png)

"This week brought to you by the generally available service that you couldn't use."

**Hosts:** [Gunnar Grosch](https://sessionize.com/gunnargrosch/) (Principal Developer Advocate, [AWS](https://aws.amazon.com/)) · [AJ Stuyvenberg](https://aws.amazon.com/developer/community/heroes/aj-stuyvenberg/) (Staff Engineer, [Datadog](https://www.datadoghq.com/)) · [Matthew Bonig](https://about.me/matthew.bonig) (CDK specialist) · [Danielle Heberling](https://aws.amazon.com/developer/community/heroes/danielle-heberling/) (software engineer, FedRAMP/gov) · [Chris Williams](https://aws.amazon.com/developer/community/heroes/chris-williams/) (Cloud Therapist, [vBrownBag](https://vbrownbag.com/))

---

## [0:52] Lambda MicroVMs: The Most Fun Product You Can't Justify

![The panel discusses Lambda MicroVMs](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/group-intro.png)

**AJ Stuyvenberg:** I'm actually really excited about [Lambda MicroVMs](https://aws.amazon.com/lambda/lambda-microvms/) from a technological standpoint. It's their most creative compute product that the serverless team has offered in years. What it is is a hosted version of Lambda's underlying VM technology called [Firecracker](https://firecracker-microvm.github.io/) — not tied to a function response lifecycle, more flexible than [Fargate](https://aws.amazon.com/fargate/). They give you an actual running instance that you can connect to, send data to, get responses back in an interactive session, and suspend and resume as necessary.

When you deploy one, it snapshots the running micro VM so it restores the contents of RAM right back to the running machine. It comes up in milliseconds.

Where the complaints start — the pricing model gives AWS's own teams preferential pricing over external customers. [Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/), built on top of micro VMs, charges only for active CPU time — the moments the Linux scheduler has your task on-core. External customers using raw MicroVMs pay for the full session including I/O wait.

I looked at traces from a production service at [Datadog](https://www.datadoghq.com/), and 0.5% of the time is active CPU time. The rest is waiting for LLM inference. So if you're doing pure agent development, this is probably not the product for you. It's roughly 2.5 to 3.5x more expensive than Fargate.

In the state that it is in right now, who is it for? I have no idea. That's the part that I cannot parse. It's like the most fun I've had with a Lambda branded product in a long time. It feels like the first time I tried Heroku. But I just don't know where it lands in 2026.

![Danielle Heberling on the FedRAMP angle](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/danielle-speaking.png)

**Danielle Heberling:** For me it looks awesome, but still struggling to find the use case. I may consider it for runners — I use [CodeBuild](https://aws.amazon.com/codebuild/) runners for GitHub and occasionally it takes time to start up. For my company, we need to be FedRAMP certified because we build stuff for state governments. All self-hosted runners all the way. This is another option aside from vanilla ECS Fargate versus CodeBuild.

---

## [13:43] CloudFormation Express Mode: It Lies to You

![The panel's reaction when they learn Express Mode disables rollback](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/cfn-group-reaction.png)

**Matthew Bonig:** [CloudFormation Express Mode](https://aws.amazon.com/about-aws/whats-new/2026/06/aws-cloudformation-cdk/) claims you speed up deployments by four times. The basis is they no longer wait for resource stabilization before they say the stack is complete.

That's not a finished deployment. They're just cheating. The resources aren't available. You can't hit them. This breaks a fundamental contract that has been in CloudFormation since day one: if the stack is done, the resources are available.

**Gunnar:** So basically it reports done before it's actually done.

**Matthew:** It lies. It lies to you.

**Danielle Heberling:** I would also caution listeners that I was reading in the docs that it disables rollback by default.

**Chris Williams:** What? What the hell? Why?

**Matthew:** Because they're no longer watching for stabilization. They don't know that they have to roll back.

**Chris:** What the hell is the actual point of this then?

**Chris:** Perhaps the team got tired of everybody saying [Terraform](https://www.terraform.io/) completes ten times faster. And they were like, screw that, we're gonna do it. Just not correctly.

**Matthew:** If you want faster infrastructure deployments, the answer is faster services, not lying about completion. The CloudFormation team gets blamed, but it's not their fault RDS takes 20 minutes to spin up. They probably have one of the hardest jobs in all of AWS because their performance is completely dependent upon everybody else in the organization.

---

## [25:01] Cloudflare Temporary Accounts: Why Don't More People Copy This?

**Danielle Heberling:** This one is less yelling and more "this is pretty awesome, why don't more people copy this?" [Cloudflare](https://www.cloudflare.com/) added a `--temporary` flag to their [Wrangler CLI](https://blog.cloudflare.com/temporary-accounts/). You do `npm wrangler deploy --temporary` — it creates a temporary Cloudflare account, deploys to it, gives you a claim URL. You have 60 minutes to sign up and claim it as your actual account.

I think this is a great use case for agents. If you're security-conscious and don't want to give Cloudflare credentials to your agent, this is a lower-barrier way to enable it to deploy something you can actually test without giving it your credentials.

Also awesome for students — no credit card required. Amazing for that demographic.

**Matthew Bonig:** This feels like a way of artificially propping up their new account numbers. Growth hacking.

**AJ:** The neo clouds are so good at this. Not just Cloudflare — [Vercel](https://vercel.com/) continually does this. Vercel built their fluid compute primitive on top of Lambda and sold it before Lambda could do their managed instances. It's genuinely good for AWS to have that competition.

**AJ:** And this ties to my story — I have a terrible customer reputation score in the AWS platform. When I was a student in 2012, I spun up a Microsoft SQL Server, forgot about it, got a $300 bill — more than my rent at the time — begged them to forgive it, they did. And I have the same account to this day. Every time they announce a new service, I can't access it. I had eight gigabytes of RAM total across all my MicroVMs because my C-score was so low. Mark Brooker, VP Distinguished Engineer, told me "that's between you and God." I'm in the Hero program, I've signed no less than six NDAs with AWS, and still every new service = call support.

The way to do this is the no-credit-card, free-to-try sandbox that Cloudflare's pushing. It's so much better for students than anything else.

---

## [34:46] AWS WAF: Charging AI Crawlers Per Request

![Chris Williams on WAF AI monetization](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/waf-chris.png)

**Chris Williams:** AI crawler traffic is now up to 50% or more of all internet traffic. Click-through rates are dropping, CPCs and CPMs are cratering. [AWS WAF](https://aws.amazon.com/blogs/aws/aws-waf-adds-ai-traffic-monetization-capability-to-help-content-owners-charge-ai-bots-for-content-access/) shipped an AI traffic monetization feature — lets publishers charge AI bots per request instead of just blocking them. When the bot hits protected content, WAF returns a 402 payment required. Currently uses stablecoin via Coinbase, adding Stripe integration.

You can classify GPTBot, Claude Web, do verified and unverified tiers, set reactive actions. If you want [Perplexity](https://www.perplexity.ai/) to scrape your site for a certain price, you set it up.

**Matthew:** Too little, too late. This should have been done three years ago.

**AJ:** I think this is one of the more pragmatic things AWS has shipped. It's not trying to stop crawling. That ship has sailed. It's saying given that crawling is happening, let's give site operators tools to participate in the economics.

But keep in mind — for a lot of people, agents are running from a residential IP. The same thing that makes ClaudeBot work great on a Mac mini that doesn't work well inside an AWS VM is that it looks like your traffic from your house. Your residential IP is so well fingerprinted that the ability for some WAF to figure out it's not a user and in fact an AI bot will be very, very hard.

---

## [42:16] Microsoft Frontier Co & AWS Forward-Deployed Engineering: Is Enterprise AI Stuck?

**Gunnar Grosch:** [Microsoft](https://blogs.microsoft.com/blog/2026/07/02/microsoft-frontier-company-ai-engineering-that-amplifies-and-protects-your-intelligence/) dropped two and a half billion dollars on a 6,000-person subsidiary called Frontier Co. [AWS countered](https://aws.amazon.com/blogs/apn/introducing-forward-deployed-engineering-for-partners-winning-the-future-of-enterprise-ai/) with a billion-plus for forward-deployed engineering. These hyperscalers are now sending their own engineers to sit in customer offices and build the apps. This is more than ProServe. The uncomfortable part is that if AI tools require a hyperscaler to send engineers to get it working, they don't really have a platform — that's a consulting gig.

**Matthew Bonig:** Professional services people come in, become experts, write code the org can't write itself, then leave. That organization has to maintain code long-term that they didn't have the expertise to build in the first place. Either those contracts go on forever, or the org collapses under maintenance they can't handle.

There's also an inherent conflict of interest. If they're implementing AI on behalf of AWS, it behooves them to make the most token-inefficient agent they can. A lot of why companies aren't adopting AI is because there's no way to know how effective these things are. We've seen companies saying "we dumped a ton of money into AI and don't know if we have anything to show for it." Now hyperscalers who are heavily invested CapEx-wise are going "we have to fix this." They're panicking.

Maybe I'm just salty because I'm seeing a lot of people I know get laid off because of AI. It was supposed to take their jobs and it did. And now suddenly they've got nothing to go to unless they can pass a magic AI HR bot to get hired.

**Chris Williams:** I wouldn't trust somebody walking in my front door from AWS saying "I'm here to make your life easier by creating an AI bot to help you spend more money with us." They don't say that last part out loud. It's why I always go third-party with ProServe — a different company won't tell you you need multiple oversized RDS instances.

---

## [52:33] Burst Mode

![Burst Mode rapid-fire segment](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/burst-mode-panel.png)

### Claude Sonnet 5: "Available"

**Danielle Heberling:** There was an announcement about [Claude Sonnet 5 on Bedrock](https://aws.amazon.com/blogs/machine-learning/introducing-claude-sonnet-5-on-aws-anthropics-most-capable-sonnet-model/). The wording just says "is now available." From my personal experience, I tried to use it and got an access denied warning, then it asked me to contact AWS sales. I'm not sure what "available" means.

**Gunnar:** The wording is interesting because we've had a lot of instances recently where they post that something is generally available — it's not really generally available though.

**Matthew:** Generally, it is. For people with better C-scores than AJ.

**AJ:** Welcome to the permitted underclass.

### [IAM Identity Center](https://aws.amazon.com/about-aws/whats-new/2026/06/aws-iam-identity-center-account-access-customer-managed-apps/) Programmatic Access

**Matthew Bonig:** AWS changed IAM Identity Center — now there's programmatic access. This may open it up for someone to create a better version of the UI console. I'm looking forward to seeing what people create. We're gonna look at it internally to see if we can start creating interesting tooling.

**Gunnar:** The question is why did it take so long?

**Matthew:** Like a lot of things with IAM Identity Center — it's so core to your access, they have to be incredibly careful. You're gonna kill everyone's access if you get one bug.

### [Anthropic/Amazon Token Pricing](https://thenextweb.com/news/amazon-anthropic-token-pricing-openai-alternative)

**AJ Stuyvenberg:** According to The Information, Anthropic and AWS are experiencing a rift as they renegotiate their next contract. Today Amazon pays for compute hours from Anthropic. Next they're switching to per-token pricing — what everybody else pays. This is the first I've heard that AWS had a special compute-hour deal.

Brian Armstrong, CEO of Coinbase, had a viral tweet about using open-weight models — increasing token volume by an order of magnitude while decreasing cost by an order of magnitude. AI gateways are starting to use routing algorithms to determine where to send requests based on complexity.

### [58:17] Secretary of Energy Gets Booed at AWS DC Summit

**Chris Williams:** I was at the AWS DC Summit. The Secretary of Energy, Chris Wright, was David Levy's guest speaker. He came on stage and was resoundingly, roundly booed. When he said things like "building AI data centers is not gonna be materially impactful on our energy or water consumption" — everybody was calling BS. The crowd was like, bullshit.

I'm in the camp that adding 30 to 40 kilowatt-per-hour racks for a data center with 6,000 racks is going to be materially impactful no matter where you put it. It was interesting to see how many people at an AWS Summit — people who've drunk the Kool-Aid, who believe in cloud — are not having it with the energy claims.

### [1:01:12] Multi-Agent Systems Underperform Their Best Member

**Gunnar Grosch:** [Apple and Stanford researchers](https://arxiv.org/abs/2602.01011) put LLM agents into teams and gave them classic coordination tasks. They asked whether the team beat its smartest member. The teams didn't.

What's clear is that these teams don't ignore the expert — they average the expert's answer with everyone else's. It's not disagreement, it's dilution. These agents would rather be agreeable than correct. They tried labeling the expert explicitly, aggressively optimized prompts — still underperformed by 6 to 41%.

The same consensus-seeking that kills expertise also makes teams robust to sabotage. A bad actor barely mattered. One knob controls all of it.

**Chris:** Humans figured this out decades ago. The larger the group making a decision — death by consensus.

**Gunnar:** Maybe multi-agent systems is just another thing to get people to adopt AI.

**Chris:** Or burn tokens.

---

![Closing](https://dev.clouddelnorte.org/_previews/yells-at-cloud-transcript-ep-1-generally-available/screenshots/closing.png)

**Gunnar Grosch:** That is episode one of Yells at Cloud. We made it. If you want to yell at us, go to [yellsatcloudpod.com](https://yellsatcloudpod.com). Tell a coworker who has opinions about cloud and doesn't know where to put them.

---

**Production notes:** Transcribed by whisper-large-v3-turbo on AMD RX 6700 XT (ROCm) — 4m14s for 64 minutes of audio. Speaker attribution verified against [official episode transcript](https://yellsatcloudpod.com/episodes/ep-1). Pipeline: [chasko-labs/goose-cli-video-transcription-recipe](https://github.com/chasko-labs/goose-cli-video-transcription-recipe).
