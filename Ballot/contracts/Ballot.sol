pragma solidity ^0.4.11;

contract Ballot {
    
    struct Voter {
        bool voted;
        bool isAffirmVote;
    }
    
    struct Proposal {
        address addr;
        bool isEnd;
        uint number;       //投票序号
        uint totalCount;
        uint actualCount;
        uint affirmCount;
    }
    
    mapping (address => Voter) public voters;
    Proposal[] proposals;
    
    function getPostLength() view public returns (uint) {
        return proposals.length;
    }

    event Post(address sender, string message);
    
    function post() public {
        uint length = proposals.length;
        proposals.push(Proposal(msg.sender, false, length, 20, 0, 0));
        emit Post(msg.sender, "投票发布成功！");
        return;
    }
    
    event Vote(address sender, bool isSuccess, string message);
    
    function vote(uint num, bool _isAffirmVote) public {
        Voter storage voter = voters[msg.sender];
        if(voter.voted == true || num > proposals.length) {
            emit Vote(msg.sender, false, "您已经投过票啦！");
            return;
        }
        
        Proposal storage proposal = proposals[num];
        if(proposal.isEnd == true) {
            emit Vote(msg.sender, false, "该投票已结束！");
            return;
        }
        if(proposal.actualCount >= proposal.totalCount) {
            emit Vote(msg.sender, false, "投票数超过上限！");
            return;
        }
        
        voter.voted = true;
        proposal.actualCount++;
        if(_isAffirmVote == true) {
            proposal.affirmCount++;
        }
        emit Vote(msg.sender, true, "投票成功！");
        return;
    }
    
    event EndProposal(address sender, bool isSuccess, string message);
    
    function endProposal(uint num) public{
        if(num > proposals.length) {
            return;
        }
        Proposal storage proposal = proposals[num];
        if(proposal.addr != msg.sender) {
            emit EndProposal(msg.sender, false, "只有发布者才能结束投票！");
            return;
        }
        if(proposal.isEnd == true) {
            emit EndProposal(msg.sender, false, "该投票已结束！");
            return;
        }
        if(3 * proposal.actualCount > 2 * proposal.totalCount && 2 * proposal.affirmCount > proposal.actualCount) {
            proposal.isEnd = true;
            emit EndProposal(msg.sender, true, "结束投票成功，投票确认！");
        }
        else {
            proposal.isEnd = true;
            emit EndProposal(msg.sender, true, "结束投票成功，投票否决！");
        }
        return;
    }
}    

